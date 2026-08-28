#!/usr/bin/env python3
"""Reversibly replace the tested checkpoint's text-only media reminder."""
from __future__ import annotations
import argparse, hashlib, shutil
from pathlib import Path
BEFORE_SHA256="41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3"
AFTER_SHA256="15e397141077cea6619b82d0cf4b0f9419668580bebf9a82a9d1b3ad2fea3df5"
OLD="""{%- macro visible_text(content) -%}
    {%- if content is string -%}
        {{- content }}
    {%- elif content is iterable and content is not mapping -%}
        {%- for item in content -%}
            {%- if item is mapping and item.type == 'text' -%}
                {{- item.text }}
            {%- elif item is string -%}
                {{- item }}
            {%- elif item is mapping and item.type in ['image', 'image_url', 'video', 'video_url', 'audio', 'audio_url', 'input_audio'] -%}
                {%- set media_type = item.type | replace('_url', '') | replace('input_', '') -%}
                {{- \"<reminder>You are unable to process this \" ~ media_type ~ \" because you don't have multi-modal input ability. Try different methods.</reminder>\" }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{- content }}
    {%- endif -%}
{%- endmacro -%}"""
NEW="""{%- macro emit_image() -%}<|begin_of_image|><|image|><|end_of_image|>{%- endmacro -%}
{%- macro emit_video() -%}<|begin_of_video|><|video|><|end_of_video|>{%- endmacro -%}
{%- macro emit_audio() -%}<|begin_of_audio|><|end_of_audio|>{%- endmacro -%}
{%- macro visible_text(content) -%}
    {%- if content is string -%}
        {{- content }}
    {%- elif content is iterable and content is not mapping -%}
        {%- for item in content -%}
            {%- if item is mapping and item.type == 'text' -%}
                {{- item.text }}
            {%- elif item is string -%}
                {{- item }}
            {%- elif item is mapping and item.type in ['image', 'image_url'] -%}
                {{- emit_image() -}}
            {%- elif item is mapping and item.type in ['video', 'video_url'] -%}
                {{- emit_video() -}}
            {%- elif item is mapping and item.type in ['audio', 'audio_url', 'input_audio'] -%}
                {{- emit_audio() -}}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{- content }}
    {%- endif -%}
{%- endmacro -%}"""
def digest(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('model_dir',type=Path); ap.add_argument('--restore',action='store_true'); a=ap.parse_args()
    target=a.model_dir/'chat_template.jinja'; backup=a.model_dir/'chat_template.text-only.bak.jinja'
    if a.restore:
        if not target.is_file(): raise SystemExit(f'No patched target found: {target}')
        target_hash=digest(target.read_bytes())
        if target_hash!=AFTER_SHA256:
            raise SystemExit(f'Refusing restore over unknown target: {target_hash}\nExpected: {AFTER_SHA256}')
        if not backup.is_file(): raise SystemExit(f'No backup found: {backup}')
        backup_raw=backup.read_bytes(); backup_hash=digest(backup_raw)
        if backup_hash!=BEFORE_SHA256:
            raise SystemExit(f'Refusing invalid backup: {backup_hash}\nExpected: {BEFORE_SHA256}')
        shutil.copy2(backup,target)
        restored=digest(target.read_bytes())
        if restored!=BEFORE_SHA256: raise SystemExit(f'Restore verification failed: {restored}')
        print(f'restored={target} sha256={restored}'); return
    raw=target.read_bytes(); current=digest(raw)
    if current==AFTER_SHA256:
        if not backup.is_file() or digest(backup.read_bytes())!=BEFORE_SHA256:
            raise SystemExit('Template is patched but its tested original backup is missing or invalid; refusing to claim reversibility')
        print(f'already_patched={target} sha256={current} backup_sha256={BEFORE_SHA256}'); return
    if current!=BEFORE_SHA256: raise SystemExit(f'Refusing unknown template revision: {current}\nExpected: {BEFORE_SHA256}')
    text=raw.decode();
    if text.count(OLD)!=1: raise SystemExit('Expected exactly one text-only media block')
    shutil.copy2(target,backup); target.write_text(text.replace(OLD,NEW),encoding='utf-8'); final=digest(target.read_bytes())
    if final!=AFTER_SHA256:
        shutil.copy2(backup,target); raise SystemExit(f'Unexpected patched hash {final}; restored backup')
    print(f'patched={target} sha256={final} backup={backup}')
if __name__=='__main__':main()
