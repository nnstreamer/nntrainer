
// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   mman_windows.h
 * @date   8 April 2025
 * @brief  Windows implementation of posix mman
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Grzegorz Kisala <g.kisala@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#if defined(_WIN32)

#include "mman_windows.h"
#include <errno.h>
#include <io.h>
#include <windows.h>

namespace {

DWORD selectProtect(const int prot) {
  DWORD protect = 0;

  if (prot == PROT_NONE) {
    return protect;
  }

  if (prot & PROT_EXEC) {
    protect = (prot & PROT_WRITE) ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
  } else {
    protect = (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
  }

  return protect;
}

DWORD selectDesiredAccess(const int prot) {
  DWORD desiredAccess = 0;

  if (prot == PROT_NONE) {
    return desiredAccess;
  }

  if (prot & PROT_READ) {
    desiredAccess |= FILE_MAP_READ;
  }

  if (prot & PROT_WRITE) {
    desiredAccess |= FILE_MAP_WRITE;
  }

  if (prot & PROT_EXEC) {
    desiredAccess |= FILE_MAP_EXECUTE;
  }

  return desiredAccess;
}

} // namespace

void *mmap(void *addr, size_t len, int prot, int flags, int fd, uint64_t offset) {
  const DWORD protect = selectProtect(prot);
  const DWORD desired_access = selectDesiredAccess(prot);


  const bool anonymous = (flags & MAP_ANONYMOUS) || (fd == -1);                             
  HANDLE hFile = INVALID_HANDLE_VALUE;  // invalid handle value
  if(!anonymous){
    hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE) { 
      errno = EBADF;
      return (void*)-1;
    }
  }
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  const uint64_t gran = (uint64_t)si.dwAllocationGranularity;
  uint64_t base_off = 0;
  size_t delta = 0;
  SIZE_T map_len = len;
  
  if(!anonymous){
    base_off = offset - (offset % gran);
    delta = (size_t)(offset - base_off);
    map_len += delta;
  }else{
    base_off = 0;
    delta = 0;
    map_len = len;
  }

  uint64_t file_size = 0;
  if(!anonymous){
    LARGE_INTEGER fs;
    if(!GetFileSizeEx(hFile, &fs)){
      errno = EIO;
      return (void*)-1;
    }
    file_size = (uint64_t)fs.QuadPart;
    uint64_t end_needed = base_off + (uint64_t)map_len;

    if(end_needed > file_size){
      if(prot & PROT_WRITE){
        LARGE_INTEGER li;
        li.QuadPart = (LONGLONG)end_needed;
        if(!SetFilePointerEx(hFile, li, nullptr, FILE_BEGIN) || !SetEndOfFile(hFile)){
          errno = EIO;
          return (void*)-1;
        }
        file_size = end_needed;
      }else{
        uint64_t remain = (file_size > base_off)? (file_size - base_off):0;
        if(remain == 0){
          errno = EINVAL;
          return (void*)-1;
        }
        if((uint64_t)map_len > remain) map_len = (SIZE_T)remain;
      }
    }
  }

  uint64_t section_max=0;
  if(anonymous){
    uint64_t need = (uint64_t) map_len;
    section_max = (need + (gran -1))/gran *gran;
  }else{
    uint64_t end_needed = base_off + (uint64_t) map_len;
    if(end_needed > file_size) end_needed = file_size;
    section_max = end_needed;
  }
  
  DWORD max_size_high = (DWORD)(section_max >>32);
  DWORD max_size_low = (DWORD)(section_max & 0xFFFFFFFFu);

  ULARGE_INTEGER uoff; 
  uoff.QuadPart = base_off;
  
  errno = 0;

  if (len == 0 || prot == PROT_EXEC) {
    errno = EINVAL;
    return MAP_FAILED;
  }

  HANDLE file_mapping =
    CreateFileMapping(anonymous ? INVALID_HANDLE_VALUE : hFile, nullptr, protect, max_size_high, max_size_low, nullptr);
  if (!file_mapping) {
    errno = GetLastError();
    return MAP_FAILED;
  }
  
  void *map_view = nullptr;
  void *base_addr = nullptr;

  if (flags & MAP_FIXED) {
    if(((uintptr_t)addr % gran) != 0){
      CloseHandle(file_mapping);
      errno = EINVAL;
      return(void*)-1;
    }
      base_addr = addr;
      map_view = MapViewOfFileEx(file_mapping, desired_access, uoff.HighPart,
                               uoff.LowPart, len, base_addr);
  } else {
    map_view = MapViewOfFile(file_mapping, desired_access, uoff.HighPart,
                               uoff.LowPart, len);
  }
  if (!map_view) {
    errno = GetLastError();
    CloseHandle(file_mapping);
    return MAP_FAILED;
  }

  uint8_t*ret = (uint8_t*) map_view + delta;

  CloseHandle(file_mapping);

  return (void*)ret;
}

int munmap(void *addr, size_t len) {
  if (UnmapViewOfFile(addr)) {
    return 0;
  }

  return -1;
}

int msync(void *addr, size_t len, int flags) {
  if (FlushViewOfFile(addr, len)) {
    return 0;
  }

  return -1;
}

int mlock(const void *addr, size_t len) {
  if (VirtualLock((LPVOID)addr, len)) {
    return 0;
  }

  return -1;
}

int munlock(const void *addr, size_t len) {
  if (VirtualUnlock((LPVOID)addr, len)) {
    return 0;
  }

  return -1;
}

#endif // _WIN32
