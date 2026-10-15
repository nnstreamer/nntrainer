"""Loopback web UI and request-scoped JSON transport to the actual Android demo app."""
import argparse
import base64
import datetime
import functools
import http.server
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import threading
import time
import urllib.parse
import uuid

WEB = Path(__file__).resolve().parents[1]
RUNS = WEB / '.runtime/runs'
ACTIONS = {'status', 'select', 'before', 'train', 'after', 'run_user', 'run_all', 'lora_on', 'lora_off', 'stop'}

class BridgeError(Exception): pass

class Device:
    def __init__(self, serial, package, adb_executable="adb"):
        self.serial, self.package = serial, package
        self.adb_executable = adb_executable
        self.lock = threading.Lock()
        self.last_saved = None
        self.state_path = '/sdcard/Android/data/' + package + '/files/device_demo/state.json'

    def adb(self, *arguments, timeout=15):
        proc = subprocess.run([self.adb_executable, '-s', self.serial, *arguments], capture_output=True,
                              text=True, encoding='utf-8', errors='replace', timeout=timeout)
        if proc.returncode:
            raise BridgeError((proc.stderr or proc.stdout or 'Quick.AI 앱을 확인할 수 없습니다. 기기 앱 연결을 누르세요.').strip()[:500])
        return proc.stdout

    def read(self):
        raw = self.adb('exec-out', 'cat', self.state_path)
        try:
            state = json.loads(raw)
            assert state['schema'] == 1 and state['source'] == 'android_native' and state['package'] == self.package
            session = str(uuid.UUID(state['session_id']))
        except (ValueError, KeyError, AssertionError) as error:
            raise BridgeError('기기의 구조화 출력이 준비되지 않았습니다. 기기 앱 연결을 누르세요.') from error
        pid = self.adb('shell', 'pidof', self.package).strip().split()
        if str(state['pid']) not in pid:
            raise BridgeError('앱이 종료되어 이전 출력만 남아 있습니다. 기기 앱 연결을 누르세요.')
        key = (session, state['sequence'])
        if key != self.last_saved:
            directory = RUNS / session
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'state.json').write_text(raw, encoding='utf-8')
            with (directory / 'events.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps({k: state.get(k) for k in ['sequence','updated_at_ms','phase','message','request_id','command_status','active_user']}, ensure_ascii=False) + '\n')
            self.last_saved = key
        return {'device': state, 'serial': self.serial, 'received_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}

    def open(self):
        result = self.adb('shell', 'am', 'start', '-W', '-n', self.package + '/com.example.sampletestapp.MainActivity', '-f', '0x20000000')
        if 'Error:' in result: raise BridgeError(result[:500])
        return {'message': 'Quick.AI 앱 연결을 요청했습니다.'}

    def command(self, body):
        action, user = body.get('action'), body.get('user', 'u03')
        request_id = body.get('request_id')
        if action not in ACTIONS or user not in {'u02', 'u03'} or not isinstance(request_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',request_id):
            raise ValueError('지원하지 않는 기기 요청입니다.')
        request = {'action': action, 'user': user, 'request_id': request_id}
        encoded = base64.b64encode(json.dumps(request).encode()).decode()
        result = self.adb('shell','am','start','-W','-n',self.package+'/com.example.sampletestapp.MainActivity',
                          '-f','0x20000000','--es','quickai_demo_command_b64',encoded)
        if 'Error:' in result: raise BridgeError(result[:500])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = self.read()
            state = response['device']
            if state.get('rejected_request_id') == request_id: raise BridgeError(state['message'])
            if state.get('ack_request_id') == request_id or action == 'stop': return response
            time.sleep(.1)
        raise BridgeError('기기가 요청을 확인하지 못했습니다. 최신 데모 앱인지 확인하세요.')

class Handler(http.server.SimpleHTTPRequestHandler):
    def local(self):
        host = urllib.parse.urlsplit('http://' + self.headers.get('Host','')).hostname
        return host in {'localhost','127.0.0.1','[::1]','::1'}

    def reply(self, code, value):
        raw = json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(code); self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(raw)))
        self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        if not self.local(): self.reply(403,{'error':'로컬 데모 주소로 접속하세요.'}); return
        path = urllib.parse.urlsplit(self.path).path
        try:
            if path == '/api/status': self.reply(200,{'token':self.token,'serial':self.device.serial,'mode':'android_native_json'})
            elif path == '/api/device': self.reply(200,self.device.read())
            elif path.startswith('/api/'): self.reply(404,{'error':'Unknown API'})
            else:
                if path == '/': self.path = '/demo_combined.html'
                super().do_GET()
        except (BridgeError, subprocess.TimeoutExpired, OSError) as error: self.reply(503,{'error':str(error)})

    def do_POST(self):
        origin = self.headers.get('Origin')
        if not self.local() or (origin and urllib.parse.urlsplit(origin).netloc != self.headers.get('Host')) or self.headers.get('X-Demo-Token') != self.token:
            self.reply(403,{'error':'현재 데모 페이지에서 요청하세요.'}); return
        try:
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 2048: raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body,dict): raise ValueError('JSON object required')
            path = urllib.parse.urlsplit(self.path).path
            with self.device.lock:
                if path == '/api/open': value = self.device.open()
                elif path == '/api/command': value = self.device.command(body)
                else: self.reply(404,{'error':'Unknown API'}); return
            self.reply(200,value)
        except (ValueError,BridgeError,subprocess.TimeoutExpired,OSError) as error: self.reply(409,{'error':str(error)})

def main():
    global RUNS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8001)
    parser.add_argument('--serial',default=os.environ.get('ANDROID_SERIAL'))
    parser.add_argument('--adb',default='adb',help='ADB executable path')
    parser.add_argument('--runs-dir',type=Path,default=RUNS)
    parser.add_argument('--package',default='com.example.sampletestapp')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.]+',args.package): parser.error('Invalid package')
    if not args.serial: parser.error('Specify --serial or ANDROID_SERIAL')
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+',args.serial): parser.error('Invalid serial')
    if not 0 <= args.port <= 65535: parser.error('Invalid port')
    RUNS = args.runs_dir.expanduser().resolve()
    Handler.device = Device(args.serial,args.package,args.adb)
    Handler.token = secrets.token_urlsafe(32)
    with http.server.ThreadingHTTPServer(('127.0.0.1',args.port),functools.partial(Handler,directory=str(WEB))) as server:
        print(f'Device demo: http://localhost:{server.server_port}/demo_combined.html · {args.serial}',flush=True)
        server.serve_forever()

if __name__ == '__main__': main()
