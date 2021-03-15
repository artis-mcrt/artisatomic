#!/usr/bin/env python3

import shutil
import urllib3

file_list = [
    'fe3.rrc.ls.txt', 'ni2.ptpx.txt', 'fe3.f.fs.txt', 'fe3.en.ls.txt', 'fe1.px.txt', 'he1.ptpx.ls.txt', 'o2.px.txt',
    'fe5.ptpx.txt', 'fe4.ptpx.txt', 'o2.ptpx.txt', 'o3.ptpx.txt', 'ni2.rrc.ls.txt', 'fe5.en.ls.txt', 'h1.fsa.txt',
    'fe4.f.ls.txt', 'o3.f.ls.txt', 'fe3.px.txt', 'h1.px.txt', 'fe2.f.ls.txt', 'fe3.f.ls.txt', 'fe1.rrc.ls.txt',
    'he2.fsa.txt', 'fe2.en.ls.txt', 'ni2.en.ls.txt', 'si1.en.ls.txt', 'o1.en.ls.txt', 'he1.en.ls.txt',
    'fe4.en.ls.txt', 'fe5.px.txt', 'fe2.px2.txt', 'fe2.ptpx.txt', 'fe3.ptpx.txt', 'fe1.en.ls.txt', 'n1.px.txt',
    'o2.en.ls.txt', 'o3.px.txt', 'fe5.rrc.txt', 'o1.px.txt', 'fe2.rrc.ls.txt', 'fe2.px.txt', 'fe4.rrc.txt',
    'o1.ptpx.txt', 'o4.en.ls.txt', 'fe1.ptpx.txt', 'o3.en.ls.txt', 'n1.ptpx.txt', 'fe1.f.ls.txt', 'n1.en.ls.txt',
    'fe4.px.txt'
]

for file in file_list:
    http = urllib3.PoolManager()
    url = f'http://www.astronomy.ohio-state.edu/~nahar/nahar_radiativeatomicdata/{file.split(".")[0]}/{file}'

    c = urllib3.PoolManager()

    print(f"Downloading {url}")

    with c.request('GET', url, preload_content=False) as resp, open(file, 'wb') as out_file:
        assert resp.status == 200
        shutil.copyfileobj(resp, out_file)

    resp.release_conn()