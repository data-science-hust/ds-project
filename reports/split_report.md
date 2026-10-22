# Báo cáo chia tập (split)

Sinh bởi `scripts/build_split.py`, seed **2026** (`configs/crawling/run.json`). Không ghi thời điểm chạy để hai lần chạy cho output giống hệt từng byte. Số viết theo kiểu Việt Nam (nghìn `.`, thập phân `,`); các file CSV dùng số thô.

Input (chỉ đọc):

- `data/processed/articles_clean.parquet`: SHA-256 `11fc49cfdafd9005e78e99e4e397bf0e383b1fb525b2dece4f5a2d9d17b7cb45`
- `data/interim/template_series.csv`: SHA-256 `e0ebdabf722222f01cdb1b4fc14b36d331aa03be2862725496d647ae4a367017`
- `data/interim/dedup_review.csv`: SHA-256 `07e5a07e930da6cfca1b28a7fc9574833e686c7ae736159c73d0d21442cb3c3d`
- `data/interim/dedup_pairs.csv`: SHA-256 `858ab8f1bec0e879938450abcf65a7e1275e1580bbe99ba6fafc2b6aa330d114`
- `configs/checking/dedup_decisions.csv`: SHA-256 `8070addc98c07f3f100a9047959738f5a52275f00cdb9b6d03bb6c75ff9fd729`

## 1. Pool đủ điều kiện

Đủ điều kiện = đại diện nhóm trùng (`is_dup_representative`) và không có `dedup_excluded_reason`. Mọi category khác null đều có mapping `approved` (script từ chối chạy nếu không). Không chọn được: bài có category null (nằm trong reserve với stratum `(không có category)`, `hold_reason=no_category`) và bài seen có cặp gần trùng chưa review với bài unseen (`hold_reason=pending_dedup_review_with_unseen`, mục 5); cả nhóm `split_group` chứa bài đó cũng không chọn được.

| Nguồn | Nhóm | Bài hợp lệ | Đủ điều kiện | Có sapo | Category null | Giữ chờ review | Chọn được | Vào split | Reserve |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| afamily | seen | 3.500 | 3.393 | 3.389 | 0 | 0 | 3.393 | 2.800 | 593 |
| 2sao | seen | 3.500 | 3.436 | 3.435 | 0 | 2 | 3.434 | 2.800 | 636 |
| eva | seen | 3.341 | 3.319 | 3.319 | 0 | 2 | 3.317 | 2.800 | 519 |
| kenh14 | seen | 3.500 | 3.439 | 3.428 | 4 | 0 | 3.435 | 2.800 | 639 |
| cafef | seen | 3.487 | 3.415 | 3.413 | 0 | 0 | 3.415 | 2.800 | 615 |
| thanhnien | seen | 3.460 | 3.456 | 3.448 | 0 | 1 | 3.455 | 2.800 | 656 |
| tuoitre | seen | 3.500 | 3.494 | 3.491 | 0 | 0 | 3.494 | 2.800 | 694 |
| dantri | seen | 3.498 | 3.491 | 3.466 | 0 | 5 | 3.486 | 2.800 | 691 |
| saostar | unseen | 500 | 500 | 500 | 0 | 0 | 500 | 300 | 200 |
| yan | unseen | 500 | 500 | 217 | 0 | 0 | 500 | 300 | 200 |
| vietnamnet | unseen | 500 | 500 | 494 | 0 | 0 | 500 | 300 | 200 |
| nhandan | unseen | 491 | 491 | 468 | 0 | 0 | 491 | 300 | 191 |

Tổng: 29.777 bài hợp lệ, 29.434 đủ điều kiện, 29.420 chọn được, 23.600 vào split, 5.834 reserve; 4 bài category null (kenh14: 4); 10 bài giữ chờ review dedup với bài unseen (2sao: 2, eva: 2, thanhnien: 1, dantri: 5).

## 2. Số bài theo nguồn × split

| Nguồn | train | trong đó subset human | dev | test_seen | test_unseen | Tổng | Human (subset+dev+test) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| afamily | 2.500 | 800 | 175 | 125 | 0 | 2.800 | 1.100 |
| 2sao | 2.500 | 800 | 175 | 125 | 0 | 2.800 | 1.100 |
| eva | 2.500 | 800 | 175 | 125 | 0 | 2.800 | 1.100 |
| kenh14 | 2.500 | 800 | 175 | 125 | 0 | 2.800 | 1.100 |
| cafef | 2.500 | 800 | 175 | 125 | 0 | 2.800 | 1.100 |
| thanhnien | 2.500 | 800 | 175 | 125 | 0 | 2.800 | 1.100 |
| tuoitre | 2.500 | 800 | 175 | 125 | 0 | 2.800 | 1.100 |
| dantri | 2.500 | 800 | 175 | 125 | 0 | 2.800 | 1.100 |
| saostar | 0 | 0 | 0 | 0 | 300 | 300 | 300 |
| yan | 0 | 0 | 0 | 0 | 300 | 300 | 300 |
| vietnamnet | 0 | 0 | 0 | 0 | 300 | 300 | 300 |
| nhandan | 0 | 0 | 0 | 0 | 300 | 300 | 300 |
| **Tổng** | **20.000** | **6.400** | **1.400** | **1.000** | **1.200** | **23.600** | **10.000** |

Quota theo `configs/crawling/sources.csv`: mỗi nguồn seen (8 nguồn: afamily, 2sao, eva, kenh14, cafef, thanhnien, tuoitre, dantri): 2.800 = train 2.500 (subset human 800) + dev 175 + test_seen 125; mỗi nguồn unseen (4 nguồn: saostar, yan, vietnamnet, nhandan): 300 = test_unseen 300. Phạm vi human 10.000 = subset 6.400 + dev 1.400 + test_seen 1.000 + test_unseen 1.200.

## 3. Cơ cấu category trong từng nguồn

Quota mỗi split (và subset trong train, test_unseen) được chia theo tỷ lệ category của các bài chọn được của nguồn bằng phương pháp phần dư lớn nhất (hòa thì theo tên category). Không dùng nhãn nào. Độ lệch = max theo category của |tỷ lệ trong split − tỷ lệ trong pool chọn được|, đơn vị điểm phần trăm; giới hạn do làm tròn là 1/quota (train 0,040; subset human 0,125; dev 0,571; test_seen 0,800; test_unseen 0,333; tính với quota nhỏ nhất của mỗi cột).

| Nguồn | train | subset human | dev | test_seen | test_unseen | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| afamily | 0,02 | 0,05 | 0,39 | 0,42 | – | **0,42** |
| 2sao | 0,01 | 0,06 | 0,24 | 0,47 | – | **0,47** |
| eva | 0,02 | 0,07 | 0,25 | 0,44 | – | **0,44** |
| kenh14 | 0,02 | 0,06 | 0,29 | 0,46 | – | **0,46** |
| cafef | 0,02 | 0,06 | 0,26 | 0,45 | – | **0,45** |
| thanhnien | 0,02 | 0,07 | 0,29 | 0,42 | – | **0,42** |
| tuoitre | 0,02 | 0,06 | 0,28 | 0,42 | – | **0,42** |
| dantri | 0,02 | 0,07 | 0,31 | 0,43 | – | **0,43** |
| saostar | – | – | – | – | 0,20 | **0,20** |
| yan | – | – | – | – | 0,13 | **0,13** |
| vietnamnet | – | – | – | – | 0,20 | **0,20** |
| nhandan | – | – | – | – | 0,13 | **0,13** |

### Nguồn seen: số bài theo category

| Nguồn | Category | Đủ điều kiện | Chọn được | Tỷ lệ chọn được | train | subset human | dev | test_seen | reserve |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| afamily | Du lịch - Ẩm thực | 116 | 116 | 3,42% | 85 | 27 | 6 | 4 | 21 |
| afamily | Giáo dục | 369 | 369 | 10,88% | 272 | 87 | 19 | 14 | 64 |
| afamily | Giải trí - Văn hóa | 644 | 644 | 18,98% | 475 | 152 | 33 | 24 | 112 |
| afamily | Khác | 6 | 6 | 0,18% | 4 | 1 | 1 | 0 | 1 |
| afamily | Kinh tế - Tài chính | 275 | 275 | 8,10% | 203 | 65 | 14 | 10 | 48 |
| afamily | Làm đẹp - Thời trang | 272 | 272 | 8,02% | 200 | 64 | 14 | 10 | 48 |
| afamily | Sức khỏe | 393 | 393 | 11,58% | 290 | 93 | 20 | 15 | 68 |
| afamily | Thời sự - Xã hội | 500 | 500 | 14,74% | 368 | 118 | 26 | 18 | 88 |
| afamily | Đời sống - Gia đình | 818 | 818 | 24,11% | 603 | 193 | 42 | 30 | 143 |
| 2sao | Công nghệ - Khoa học | 176 | 176 | 5,13% | 128 | 41 | 9 | 7 | 32 |
| 2sao | Giải trí - Văn hóa | 751 | 750 | 21,84% | 546 | 175 | 38 | 27 | 140 |
| 2sao | Làm đẹp - Thời trang | 440 | 440 | 12,81% | 320 | 102 | 22 | 16 | 82 |
| 2sao | Thời sự - Xã hội | 741 | 740 | 21,55% | 539 | 172 | 38 | 27 | 137 |
| 2sao | Tử vi - Phong thủy | 608 | 608 | 17,71% | 443 | 142 | 31 | 22 | 112 |
| 2sao | Đời sống - Gia đình | 720 | 720 | 20,97% | 524 | 168 | 37 | 26 | 133 |
| eva | Du lịch - Ẩm thực | 304 | 304 | 9,16% | 229 | 73 | 16 | 12 | 47 |
| eva | Giải trí - Văn hóa | 346 | 346 | 10,43% | 261 | 83 | 18 | 13 | 54 |
| eva | Khác | 229 | 229 | 6,90% | 173 | 55 | 12 | 9 | 35 |
| eva | Làm đẹp - Thời trang | 314 | 314 | 9,47% | 237 | 76 | 17 | 12 | 48 |
| eva | Sức khỏe | 266 | 266 | 8,02% | 200 | 64 | 14 | 10 | 42 |
| eva | Thời sự - Xã hội | 858 | 856 | 25,81% | 645 | 207 | 45 | 32 | 136 |
| eva | Tử vi - Phong thủy | 194 | 194 | 5,85% | 146 | 47 | 10 | 7 | 31 |
| eva | Đời sống - Gia đình | 808 | 808 | 24,36% | 609 | 195 | 43 | 30 | 126 |
| kenh14 | (không có category) | 4 | 0 | – | 0 | 0 | 0 | 0 | 4 |
| kenh14 | Công nghệ - Khoa học | 206 | 206 | 6,00% | 150 | 48 | 10 | 8 | 38 |
| kenh14 | Du lịch - Ẩm thực | 171 | 171 | 4,98% | 124 | 40 | 9 | 6 | 32 |
| kenh14 | Giáo dục | 234 | 234 | 6,81% | 170 | 55 | 12 | 9 | 43 |
| kenh14 | Giải trí - Văn hóa | 527 | 527 | 15,34% | 384 | 123 | 27 | 19 | 97 |
| kenh14 | Khác | 7 | 7 | 0,20% | 5 | 2 | 0 | 0 | 2 |
| kenh14 | Kinh tế - Tài chính | 173 | 173 | 5,04% | 126 | 40 | 9 | 6 | 32 |
| kenh14 | Làm đẹp - Thời trang | 117 | 117 | 3,41% | 85 | 27 | 6 | 4 | 22 |
| kenh14 | Sức khỏe | 422 | 422 | 12,29% | 307 | 98 | 22 | 15 | 78 |
| kenh14 | Thế giới | 331 | 331 | 9,64% | 241 | 77 | 17 | 12 | 61 |
| kenh14 | Thể thao | 225 | 225 | 6,55% | 164 | 52 | 11 | 8 | 42 |
| kenh14 | Thời sự - Xã hội | 598 | 598 | 17,41% | 435 | 139 | 30 | 22 | 111 |
| kenh14 | Đời sống - Gia đình | 424 | 424 | 12,34% | 309 | 99 | 22 | 16 | 77 |
| cafef | Bất động sản | 367 | 367 | 10,75% | 269 | 86 | 19 | 14 | 65 |
| cafef | Công nghệ - Khoa học | 165 | 165 | 4,83% | 121 | 39 | 8 | 6 | 30 |
| cafef | Khác | 37 | 37 | 1,08% | 27 | 9 | 2 | 1 | 7 |
| cafef | Kinh tế - Tài chính | 1.112 | 1.112 | 32,56% | 814 | 260 | 57 | 41 | 200 |
| cafef | Thế giới | 312 | 312 | 9,14% | 228 | 73 | 16 | 11 | 57 |
| cafef | Thời sự - Xã hội | 306 | 306 | 8,96% | 224 | 72 | 16 | 11 | 55 |
| cafef | Đời sống - Gia đình | 1.116 | 1.116 | 32,68% | 817 | 261 | 57 | 41 | 201 |
| thanhnien | Bất động sản | 31 | 31 | 0,90% | 22 | 7 | 2 | 1 | 6 |
| thanhnien | Công nghệ - Khoa học | 168 | 168 | 4,86% | 122 | 39 | 8 | 6 | 32 |
| thanhnien | Du lịch - Ẩm thực | 53 | 53 | 1,53% | 38 | 12 | 3 | 2 | 10 |
| thanhnien | Giáo dục | 189 | 189 | 5,47% | 137 | 44 | 10 | 7 | 35 |
| thanhnien | Giải trí - Văn hóa | 282 | 282 | 8,16% | 204 | 65 | 14 | 10 | 54 |
| thanhnien | Khác | 46 | 46 | 1,33% | 33 | 11 | 2 | 2 | 9 |
| thanhnien | Kinh tế - Tài chính | 406 | 406 | 11,75% | 294 | 94 | 21 | 15 | 76 |
| thanhnien | Làm đẹp - Thời trang | 84 | 84 | 2,43% | 61 | 20 | 4 | 3 | 16 |
| thanhnien | Pháp luật | 88 | 88 | 2,55% | 64 | 20 | 4 | 3 | 17 |
| thanhnien | Sức khỏe | 196 | 196 | 5,67% | 142 | 45 | 10 | 7 | 37 |
| thanhnien | Thế giới | 330 | 330 | 9,55% | 239 | 76 | 17 | 12 | 62 |
| thanhnien | Thể thao | 438 | 438 | 12,68% | 317 | 102 | 22 | 16 | 83 |
| thanhnien | Thời sự - Xã hội | 891 | 890 | 25,76% | 644 | 206 | 45 | 32 | 170 |
| thanhnien | Xe | 46 | 46 | 1,33% | 33 | 11 | 2 | 2 | 9 |
| thanhnien | Đời sống - Gia đình | 208 | 208 | 6,02% | 150 | 48 | 11 | 7 | 40 |
| tuoitre | Bất động sản | 75 | 75 | 2,15% | 54 | 17 | 4 | 3 | 14 |
| tuoitre | Công nghệ - Khoa học | 107 | 107 | 3,06% | 77 | 25 | 5 | 4 | 21 |
| tuoitre | Du lịch - Ẩm thực | 95 | 95 | 2,72% | 68 | 22 | 5 | 3 | 19 |
| tuoitre | Giáo dục | 165 | 165 | 4,72% | 118 | 38 | 8 | 6 | 33 |
| tuoitre | Giải trí - Văn hóa | 325 | 325 | 9,30% | 233 | 74 | 16 | 12 | 64 |
| tuoitre | Khác | 95 | 95 | 2,72% | 68 | 22 | 5 | 3 | 19 |
| tuoitre | Kinh tế - Tài chính | 449 | 449 | 12,85% | 321 | 103 | 22 | 16 | 90 |
| tuoitre | Pháp luật | 230 | 230 | 6,58% | 165 | 53 | 12 | 8 | 45 |
| tuoitre | Sức khỏe | 153 | 153 | 4,38% | 109 | 35 | 8 | 6 | 30 |
| tuoitre | Thế giới | 396 | 396 | 11,33% | 283 | 91 | 20 | 14 | 79 |
| tuoitre | Thể thao | 443 | 443 | 12,68% | 317 | 101 | 22 | 16 | 88 |
| tuoitre | Thời sự - Xã hội | 762 | 762 | 21,81% | 545 | 174 | 38 | 27 | 152 |
| tuoitre | Xe | 88 | 88 | 2,52% | 63 | 20 | 4 | 3 | 18 |
| tuoitre | Đời sống - Gia đình | 111 | 111 | 3,18% | 79 | 25 | 6 | 4 | 22 |
| dantri | Bất động sản | 105 | 105 | 3,01% | 75 | 24 | 5 | 4 | 21 |
| dantri | Công nghệ - Khoa học | 140 | 140 | 4,02% | 100 | 32 | 7 | 5 | 28 |
| dantri | Du lịch - Ẩm thực | 91 | 91 | 2,61% | 65 | 21 | 5 | 3 | 18 |
| dantri | Giáo dục | 170 | 170 | 4,88% | 122 | 39 | 8 | 6 | 34 |
| dantri | Giải trí - Văn hóa | 133 | 133 | 3,82% | 95 | 30 | 7 | 5 | 26 |
| dantri | Khác | 2 | 2 | 0,06% | 1 | 0 | 0 | 0 | 1 |
| dantri | Kinh tế - Tài chính | 362 | 361 | 10,36% | 259 | 83 | 18 | 13 | 72 |
| dantri | Làm đẹp - Thời trang | 13 | 13 | 0,37% | 9 | 3 | 1 | 1 | 2 |
| dantri | Pháp luật | 375 | 375 | 10,76% | 269 | 86 | 19 | 13 | 74 |
| dantri | Sức khỏe | 178 | 178 | 5,11% | 128 | 41 | 9 | 6 | 35 |
| dantri | Thế giới | 369 | 369 | 10,59% | 265 | 85 | 18 | 13 | 73 |
| dantri | Thể thao | 302 | 301 | 8,63% | 216 | 69 | 15 | 11 | 60 |
| dantri | Thời sự - Xã hội | 991 | 988 | 28,34% | 709 | 227 | 50 | 35 | 197 |
| dantri | Xe | 107 | 107 | 3,07% | 77 | 25 | 5 | 4 | 21 |
| dantri | Đời sống - Gia đình | 153 | 153 | 4,39% | 110 | 35 | 8 | 6 | 29 |

### Nguồn unseen: số bài theo category

| Nguồn | Category | Đủ điều kiện | Chọn được | Tỷ lệ chọn được | test_unseen | reserve |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| saostar | Công nghệ - Khoa học | 11 | 11 | 2,20% | 7 | 4 |
| saostar | Du lịch - Ẩm thực | 12 | 12 | 2,40% | 7 | 5 |
| saostar | Giáo dục | 4 | 4 | 0,80% | 3 | 1 |
| saostar | Giải trí - Văn hóa | 329 | 329 | 65,80% | 197 | 132 |
| saostar | Kinh tế - Tài chính | 14 | 14 | 2,80% | 8 | 6 |
| saostar | Làm đẹp - Thời trang | 7 | 7 | 1,40% | 4 | 3 |
| saostar | Thế giới | 29 | 29 | 5,80% | 17 | 12 |
| saostar | Thể thao | 38 | 38 | 7,60% | 23 | 15 |
| saostar | Thời sự - Xã hội | 56 | 56 | 11,20% | 34 | 22 |
| yan | Giải trí - Văn hóa | 325 | 325 | 65,00% | 195 | 130 |
| yan | Khác | 94 | 94 | 18,80% | 56 | 38 |
| yan | Làm đẹp - Thời trang | 81 | 81 | 16,20% | 49 | 32 |
| vietnamnet | Bất động sản | 10 | 10 | 2,00% | 6 | 4 |
| vietnamnet | Công nghệ - Khoa học | 26 | 26 | 5,20% | 16 | 10 |
| vietnamnet | Du lịch - Ẩm thực | 9 | 9 | 1,80% | 5 | 4 |
| vietnamnet | Giáo dục | 27 | 27 | 5,40% | 16 | 11 |
| vietnamnet | Giải trí - Văn hóa | 48 | 48 | 9,60% | 29 | 19 |
| vietnamnet | Khác | 11 | 11 | 2,20% | 7 | 4 |
| vietnamnet | Kinh tế - Tài chính | 59 | 59 | 11,80% | 35 | 24 |
| vietnamnet | Pháp luật | 32 | 32 | 6,40% | 19 | 13 |
| vietnamnet | Sức khỏe | 32 | 32 | 6,40% | 19 | 13 |
| vietnamnet | Thế giới | 31 | 31 | 6,20% | 19 | 12 |
| vietnamnet | Thể thao | 70 | 70 | 14,00% | 42 | 28 |
| vietnamnet | Thời sự - Xã hội | 108 | 108 | 21,60% | 65 | 43 |
| vietnamnet | Xe | 16 | 16 | 3,20% | 10 | 6 |
| vietnamnet | Đời sống - Gia đình | 21 | 21 | 4,20% | 12 | 9 |
| nhandan | Công nghệ - Khoa học | 17 | 17 | 3,46% | 10 | 7 |
| nhandan | Du lịch - Ẩm thực | 8 | 8 | 1,63% | 5 | 3 |
| nhandan | Giáo dục | 11 | 11 | 2,24% | 7 | 4 |
| nhandan | Giải trí - Văn hóa | 30 | 30 | 6,11% | 18 | 12 |
| nhandan | Khác | 10 | 10 | 2,04% | 6 | 4 |
| nhandan | Kinh tế - Tài chính | 52 | 52 | 10,59% | 32 | 20 |
| nhandan | Pháp luật | 17 | 17 | 3,46% | 10 | 7 |
| nhandan | Sức khỏe | 10 | 10 | 2,04% | 6 | 4 |
| nhandan | Thế giới | 47 | 47 | 9,57% | 29 | 18 |
| nhandan | Thể thao | 21 | 21 | 4,28% | 13 | 8 |
| nhandan | Thời sự - Xã hội | 268 | 268 | 54,58% | 164 | 104 |

## 4. Sapo theo split

Dev bắt buộc có sapo (mục 3.1: so sánh title / title+sapo trên cùng bài); quota dev theo category được lấy chỉ từ bài có sapo. Train, subset và test không yêu cầu sapo.

| Split | Bài | Có sapo | Tỷ lệ |
| --- | ---: | ---: | ---: |
| train | 20.000 | 19.965 | 99,83% |
| subset human | 6.400 | 6.392 | 99,88% |
| dev | 1.400 | 1.400 | 100,00% |
| test_seen | 1.000 | 992 | 99,20% |
| test_unseen | 1.200 | 1.018 | 84,83% |
| reserve | 5.834 | 5.693 | 97,58% |

## 5. split_group (đơn vị lấy mẫu) và cặp gần trùng chưa gộp

Bài cùng `dup_group` đã gộp thành một đại diện ở bước clean. Các cặp gần trùng mà bước clean liệt kê nhưng không gộp (chuỗi template cùng nguồn; cặp `dedup_review.csv` chưa có quyết định; liên kết bị chặn trong `dedup_pairs.csv`) được nối (union-find) thành một `split_group` nếu hai bài cùng phía seen/unseen, để hai tiêu đề gần giống hệt không bao giờ nằm ở hai split. Theo mặc định của nhóm, các bài này được giữ lại, không xóa; nối nhóm chỉ ràng buộc chúng vào cùng split. Cặp seen–unseen không thể chung split: nếu chưa có quyết định `separate` trong `configs/checking/dedup_decisions.csv`, bài seen bị giữ trong reserve (`hold_reason=pending_dedup_review_with_unseen`) cho tới khi review dedup quyết định (`link` thì bước clean loại bài seen như `duplicate_of_unseen`). Quota tính theo bài; nhóm không vừa phần quota còn lại của mọi ô (nguồn, category) mà nó chạm tới thì bị bỏ qua, xét nhóm tiếp theo theo thứ tự seed.

| Loại cặp | Dòng trong file | Cặp chưa gộp | Nối vào split_group | Seen–unseen: giữ bài seen | Seen–unseen: đã `separate` | Đã quyết định `separate` | Hai bài ở hai split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chuỗi template cùng nguồn (`template_series.csv`) | 521 | 521 | 521 | 0 | 0 | 0 | 0 |
| Cặp cần review (`dedup_review.csv`) | 209 | 174 | 165 | 9 | 0 | 0 | 0 |
| Liên kết bị chặn (`dedup_pairs.csv`, applied=False) | 2 | 2 | 2 | 0 | 0 | 2 | 0 |

Một cặp nằm ở nhiều file được tính một lần, theo thứ tự template → review → bị chặn. Cột "Hai bài ở hai split" chỉ được khác 0 với cặp đã quyết định `separate` (kiểm tra ở mục 7). Dòng `dedup_review.csv` của liên kết đã áp dụng (`auto_link_check`, `exact_date_gap`) không phải cặp chưa gộp.

- 521 cặp template trong 91 chuỗi; 29.135 split_group trong pool, trong đó **160 nhóm ≥2 bài** (459 bài).
- Kích thước nhóm nhiều bài: 2 bài: 124, 3 bài: 20, 4 bài: 6, 5 bài: 3, 6 bài: 1, 7 bài: 1, 8 bài: 1, 9 bài: 1, 16 bài: 1, 24 bài: 1, 42 bài: 1.
- Nhóm nhiều bài trải nhiều category: 36; trải nhiều nguồn: 88.
- `dup_group` seen bị giữ chờ review với bài unseen: 9.
- Nơi nhận nhóm nhiều bài: dev: 5 nhóm/12 bài, reserve: 21 nhóm/59 bài, test_seen: 11 nhóm/29 bài, test_unseen: 1 nhóm/2 bài, train: 122 nhóm/357 bài.

## 6. Reserve

`data/manifests/reserve_list.csv`: mọi bài đủ điều kiện không vào split, kèm `has_sapo` và `hold_reason`. `reserve_rank` tiếp nối thứ tự seed của từng (nguồn, category) (1 = bài thay thế đầu tiên). Dùng để thay bài bị loại khi gán nhãn (mục 2.4): lấy bài có `reserve_rank` nhỏ nhất cùng nguồn/category và `hold_reason` rỗng; **thay bài dev thì bài thay thế phải có `has_sapo=True`** (cả `split_group` của nó), bỏ qua bài không có sapo. Nếu bài thay thế thuộc `split_group` nhiều bài thì phải lấy cả nhóm hoặc bỏ qua nhóm đó. Không chọn theo nhãn LLM hay kết quả model. Số bài reserve theo nguồn:

| Nguồn | Reserve | Có sapo | Có hold_reason | Dùng được | Category ít reserve dùng được nhất |
| --- | ---: | ---: | ---: | ---: | --- |
| afamily | 593 | 592 | 0 | 593 | Khác: 1 |
| 2sao | 636 | 636 | 2 | 634 | Công nghệ - Khoa học: 32 |
| eva | 519 | 519 | 2 | 517 | Tử vi - Phong thủy: 31 |
| kenh14 | 639 | 636 | 4 | 635 | Khác: 2 |
| cafef | 615 | 615 | 0 | 615 | Khác: 7 |
| thanhnien | 656 | 656 | 1 | 655 | Bất động sản: 6 |
| tuoitre | 694 | 692 | 0 | 694 | Bất động sản: 14 |
| dantri | 691 | 686 | 5 | 686 | Khác: 1 |
| saostar | 200 | 200 | 0 | 200 | Giáo dục: 1 |
| yan | 200 | 88 | 0 | 200 | Làm đẹp - Thời trang: 32 |
| vietnamnet | 200 | 196 | 0 | 200 | Bất động sản: 4 |
| nhandan | 191 | 177 | 0 | 191 | Du lịch - Ẩm thực: 3 |

Số reserve theo từng category: cột reserve ở mục 3 và `reports/split_counts.csv`.

## 7. Kiểm tra (đều đạt)

- Số bài đúng quota theo nguồn × split và subset human
- Mỗi bài đủ điều kiện nằm đúng một lần trong manifest hoặc reserve
- Mỗi split_group/dup_group chỉ thuộc một split (không tách giữa split và reserve)
- Không bài nào trong train/dev/test_seen chung dup_group/split_group với bài unseen (bản seen trùng unseen đã bị loại ở bước clean)
- Không cặp gần trùng chưa gộp nào (chuỗi template, `dedup_review.csv`, liên kết bị chặn trong `dedup_pairs.csv`; trừ cặp đã quyết định `separate`) có hai bài ở hai split khác nhau
- Bài seen có cặp gần trùng chưa review với bài unseen không vào split nào (giữ trong reserve)
- Subset human nằm trong train
- Mọi bài dev có sapo; cột has_sapo của reserve khớp bảng clean
- Nguồn unseen chỉ ở test_unseen, nguồn seen không ở test_unseen
- Bảng blind và annotation_map phủ đúng phạm vi human, annotation_id duy nhất
- Xác định: build hai lần từ cùng input cho output giống hệt từng byte (khác thì script dừng, không ghi file)

## 8. Bảng blind và ánh xạ

- `data/annotations/blind/blind_dev.csv`: 1.400 dòng (dev; gán nhãn trước theo flow của nhóm).
- `data/annotations/blind/blind_train_subset_test.csv`: 8.600 dòng (subset train + test_seen + test_unseen).
- Hai bảng chỉ có `annotation_id,title`; không có nguồn, URL, split, category hay nhãn LLM. Thứ tự dòng là một hoán vị seed trộn đều nguồn/category/split. Title giữ nguyên như clean; 1 title có xuống dòng bên trong (CSV đặt trong ngoặc kép, đọc bằng CSV reader).
- `annotation_id` = `A` + 5 chữ số lấy ngẫu nhiên (seed) từ 10000–99999, không tuần tự theo nguồn, split hay `record_id`.
- `data/manifests/annotation_map.csv` (nội bộ, không gửi annotator): 10.000 dòng `annotation_id, record_id, split, source`.

## 9. Việc còn mở

- `annotation_assignment.csv` (mục 2.2: 9 chunk C1..C9, người P1..P9, lô ~100 mẫu) do P3 (lead human annotation) tạo từ `annotation_map.csv` và hai bảng blind; bước split không tạo file này.
- Review dedup: 174 cặp `dedup_review.csv` chưa có quyết định (đã được giữ cùng split hoặc giữ bài seen trong reserve, nên không rò rỉ giữa split). Nên review (ghi `link`/`separate` vào `configs/checking/dedup_decisions.csv`) **trước khi bắt đầu gán nhãn**: chạy lại `build_clean.py` + `build_split.py` sau khi đổi quyết định sẽ đổi split.
- Quy trình thay bài từ reserve khi gán nhãn (cấp `annotation_id` mới cho bài thay thế, cập nhật manifest) chưa tự động hóa.
- Bài category null (4) chỉ nằm trong reserve; cần map category nếu muốn dùng.
