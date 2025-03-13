# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2024 Daekyoung Jung <dk.zz79ya@gmail.com>
#
# @file prepare_encoder.ps1
# @date 13 MAR 2025
# @brief This file is a helper tool to build encoder at LLM
# @author Daekyoung Jung <dk.zz79ya@gmail.com>
#
# usage: ./prepare_encoder.ps1 target version(either 0.1, 0.2)

param (
    [string]$Target,
    [string]$TargetVersion
)

$TarPrefix = "encoder"
$TarName = "$TarPrefix-$TargetVersion.tar.gz"
$Url = "https://github.com/nnstreamer/nnstreamer-android-resource/raw/main/external/$TarName"

Write-Output "PREPARING Encoder at $Target"

if (-Not (Test-Path $Target)) {
    New-Item -ItemType Directory -Path $Target
}

Push-Location $Target

function _download_encoder {
    if (Test-Path $TarName) {
        Write-Output "$TarName exists, skip downloading"
        return
    }

    Write-Output "[Encoder] downloading $TarName"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $TarName
        Write-Output "[Encoder] Finish downloading encoder"
    } catch {
        Write-Output "[Encoder] Download failed, please check url"
        exit $_
    }
}

function _untar_encoder {
    Write-Output "[Encoder] untar encoder"
    Write-Output $TarName
    Write-Output $Target
    tar -zxvf $TarName
    Remove-Item $TarName

    if ($TargetVersion -eq "0.1") {
        Move-Item -Path "ctre-unicode.hpp", "json.hpp", "encoder.hpp" -Destination "..\PicoGPT\jni\"
        Write-Output "[Encoder] Finish moving encoder to PicoGPT"
    }

    if ($TargetVersion -eq "0.2") {
        Move-Item -Path "ctre-unicode.hpp", "json.hpp", "encoder.hpp" -Destination "..\LLaMA\jni\"
        Write-Output "[Encoder] Finish moving encoder to LLaMA"
    }
}

if (-Not (Test-Path "$TarPrefix")) {
    _download_encoder
    _untar_encoder
}

Pop-Location
