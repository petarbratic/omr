"""
debug_region_extractor.py

Ručni test za extract_regions(img_bgr).

- Učita sliku (ili listu slika)
- Pozove extract_regions
- Nacrta bbox-ove i sačuva debug sliku u OUT_DIR
- Ispiše broj regiona

Podesi IMAGE_PATHS i OUT_DIR u DEFINES.
"""

import os
from typing import List

import cv2

from srcNonE2E.data.region_extractor import extract_regions


# =========================================================
# DEFINES
# =========================================================

# Možeš uneti relativno u odnosu na root projekta, ili apsolutno
from typing import List

from typing import List

from typing import List

IMAGE_PATHS: List[str] = [
    "data/primus_raw/package_ab/211006179-1_1_1/211006179-1_1_1.png",
    "data/primus_raw/package_ab/220034587-1_2_1/220034587-1_2_1.png",
    "data/primus_raw/package_ab/211006309-1_1_1/211006309-1_1_1.png",
    "data/primus_raw/package_ab/212003384-1_1_1/212003384-1_1_1.png",
    "data/primus_raw/package_ab/211006989-1_4_1/211006989-1_4_1.png",
    "data/primus_raw/package_ab/230005700-1_1_1/230005700-1_1_1.png",
    "data/primus_raw/package_aa/000125088-3_1_1/000125088-3_1_1.png",
    "data/primus_raw/package_ab/201004478-1_1_1/201004478-1_1_1.png",
    "data/primus_raw/package_ab/230003572-1_4_1/230003572-1_4_1.png",
    "data/primus_raw/package_ab/190004669-1_2_1/190004669-1_2_1.png",
    "data/primus_raw/package_ab/212001724-1_4_1/212001724-1_4_1.png",
    "data/primus_raw/package_ab/212002069-1_1_1/212002069-1_1_1.png",
    "data/primus_raw/package_ab/220030625-1_1_1/220030625-1_1_1.png",
    "data/primus_raw/package_aa/000106564-1_1_1/000106564-1_1_1.png",
    "data/primus_raw/package_ab/201009209-1_1_1/201009209-1_1_1.png",
    "data/primus_raw/package_ab/220018403-1_2_1/220018403-1_2_1.png",
    "data/primus_raw/package_aa/000119651-1_1_1/000119651-1_1_1.png",
    "data/primus_raw/package_ab/225000378-1_1_1/225000378-1_1_1.png",
    "data/primus_raw/package_ab/230004325-1_2_2/230004325-1_2_2.png",
    "data/primus_raw/package_aa/000109914-1_1_1/000109914-1_1_1.png",
    "data/primus_raw/package_ab/212003446-1_3_1/212003446-1_3_1.png",
    "data/primus_raw/package_ab/211007095-1_5_1/211007095-1_5_1.png",
    "data/primus_raw/package_ab/190009738-1_1_1/190009738-1_1_1.png",
    "data/primus_raw/package_aa/100501051-1_4_3/100501051-1_4_3.png",
    "data/primus_raw/package_ab/220017729-1_3_1/220017729-1_3_1.png",
    "data/primus_raw/package_aa/000124029-1_1_1/000124029-1_1_1.png",
    "data/primus_raw/package_ab/220017472-1_2_1/220017472-1_2_1.png",
    "data/primus_raw/package_ab/220032252-1_1_1/220032252-1_1_1.png",
    "data/primus_raw/package_ab/150200715-1_1_1/150200715-1_1_1.png",
    "data/primus_raw/package_aa/000120646-6_1_1/000120646-6_1_1.png",
    "data/primus_raw/package_ab/230005988-1_1_1/230005988-1_1_1.png",
    "data/primus_raw/package_aa/000115172-1_1_1/000115172-1_1_1.png",
    "data/primus_raw/package_aa/000102397-1_1_1/000102397-1_1_1.png",
    "data/primus_raw/package_aa/000126730-1_1_1/000126730-1_1_1.png",
    "data/primus_raw/package_aa/000120887-1_1_1/000120887-1_1_1.png",
    "data/primus_raw/package_ab/230003277-1_2_1/230003277-1_2_1.png",
    "data/primus_raw/package_ab/220010705-1_1_1/220010705-1_1_1.png",
    "data/primus_raw/package_aa/100500561-1_2_2/100500561-1_2_2.png",
    "data/primus_raw/package_ab/220018403-1_1_1/220018403-1_1_1.png",
    "data/primus_raw/package_ab/201001020-1_1_1/201001020-1_1_1.png",
    "data/primus_raw/package_ab/220001169-1_2_1/220001169-1_2_1.png",
    "data/primus_raw/package_ab/211004478-1_5_1/211004478-1_5_1.png",
    "data/primus_raw/package_ab/211006383-1_1_1/211006383-1_1_1.png",
    "data/primus_raw/package_ab/201007623-1_1_1/201007623-1_1_1.png",
    "data/primus_raw/package_aa/000106221-3_1_1/000106221-3_1_1.png",
    "data/primus_raw/package_ab/211003305-1_1_1/211003305-1_1_1.png",
    "data/primus_raw/package_aa/000107941-1_2_1/000107941-1_2_1.png",
    "data/primus_raw/package_aa/100230431-1_1_1/100230431-1_1_1.png",
    "data/primus_raw/package_ab/220010905-1_10_1/220010905-1_10_1.png",
    "data/primus_raw/package_ab/211008487-1_68_1/211008487-1_68_1.png",
    "data/primus_raw/package_ab/201008888-1_1_1/201008888-1_1_1.png",
    "data/primus_raw/package_ab/190005785-1_1_1/190005785-1_1_1.png",
    "data/primus_raw/package_ab/230002048-1_5_1/230002048-1_5_1.png",
    "data/primus_raw/package_aa/000120485-1_1_1/000120485-1_1_1.png",
    "data/primus_raw/package_ab/211003696-1_1_1/211003696-1_1_1.png",
    "data/primus_raw/package_ab/220015625-1_1_1/220015625-1_1_1.png",
    "data/primus_raw/package_ab/211010260-1_1_1/211010260-1_1_1.png",
    "data/primus_raw/package_ab/211002810-1_1_1/211002810-1_1_1.png",
    "data/primus_raw/package_aa/000110687-1_1_1/000110687-1_1_1.png",
    "data/primus_raw/package_aa/000107740-1_1_2/000107740-1_1_2.png",
    "data/primus_raw/package_ab/220031094-1_1_1/220031094-1_1_1.png",
    "data/primus_raw/package_ab/225001711-1_1_1/225001711-1_1_1.png",
    "data/primus_raw/package_aa/000120324-3_1_1/000120324-3_1_1.png",
    "data/primus_raw/package_aa/000120921-1_1_1/000120921-1_1_1.png",
    "data/primus_raw/package_aa/100500516-1_2_2/100500516-1_2_2.png",
]

OUT_DIR = "out/debug_regions"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def draw_bboxes(img_bgr, bboxes):
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def main() -> None:
    ensure_dir(OUT_DIR)

    if not IMAGE_PATHS:
        raise ValueError("Dodaj bar jednu putanju u IMAGE_PATHS u debug_region_extractor.py")

    for p in IMAGE_PATHS:
        if not os.path.exists(p):
            print(f"NE POSTOJI: {p}")
            continue

        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"NE MOGU DA UCITAM: {p}")
            continue

        bboxes = extract_regions(img)
        print(f"{p} -> regions: {len(bboxes)}")

        vis = draw_bboxes(img, bboxes)

        base = os.path.basename(p).replace(".png", "")
        out_path = os.path.join(OUT_DIR, f"{base}_boxes.png")
        cv2.imwrite(out_path, vis)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()