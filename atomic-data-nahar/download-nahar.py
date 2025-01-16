#!/usr/bin/env python3
import shutil
from pathlib import Path

import urllib3


def main():
    file_list = [
        "fe1.en.ls.txt",
        "fe1.f.ls.txt",
        "fe1.ptpx.txt",
        "fe1.px.txt",
        "fe1.rrc.txt",
        "fe2.en.ls.txt",
        "fe2.f.ls.txt",
        "fe2.ptpx.txt",
        "fe2.px.txt",
        "fe2.px.txt",
        "fe2.rrc-ls.txt",
        "fe3.en.ls.txt",
        "fe3.f.fs.txt",
        "fe3.f.ls.txt",
        "fe3.ptpx.txt",
        "fe3.px.txt",
        "fe3.rrc.ls.txt",
        "fe4.en.ls.txt",
        "fe4.f.ls.txt",
        "fe4.ptpx.txt",
        "fe4.px.txt",
        "fe4.rrc.txt",
        "fe5.en.ls.txt",
        "fe5.ptpx.txt",
        "fe5.px.txt",
        "fe5.rrc.txt",
        "h1.fsa.txt",
        "h1.px.txt",
        "he1.en.ls.txt",
        "he1.ptpx.ls.txt",
        "he2.fsa.txt",
        "n1.en.ls.txt",
        "n1.ptpx.txt",
        "n1.px.txt",
        "ni2.en.ls.txt",
        "ni2.ptpx.txt",
        "ni2.rrc.txt",
        "o1.en.ls.txt",
        "o1.ptpx.txt",
        "o1.px.txt",
        "o2.en.ls.txt",
        "o2.ptpx.txt",
        "o2.px.txt",
        "o3.en.ls.txt",
        "o3.f.ls.txt",
        "o3.ptpx.txt",
        "o3.px.txt",
        "o4.en.ls.txt",
        "si1.en.ls.txt",
    ]

    for file in file_list:
        if Path(file).exists():
            print(f"{file} already exists. Skipping.")
        else:
            url = f"https://norad.astronomy.osu.edu/{file.split('.')[0]}/{file}"

            c = urllib3.PoolManager()

            print(f"Downloading {url}")

            with c.request("GET", url, preload_content=False) as resp, open(file, "wb") as out_file:
                assert resp.status == 200
                shutil.copyfileobj(resp, out_file)

            resp.release_conn()


if __name__ == "main":
    main()
