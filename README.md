# Point it Up

Tunjuk ke arah yang sama dengan pointer yang berjalan, tanpa menggunakan dance-pad dan juga keypad.

Point it up adalah sebuah game interaktif berbasis Python yang terinspirasi dari Arcade Pump-It-Up, dimana sebuah pointer akan bergerak seiring dengan ritme lagu dan player harus menginjak/memencet keypad untuk mendapatkan skor. Pada game ini, digunakan konsep yang sama tanpa menggunakan keyboard maupun dance-pad apapun. Hanya dengan telunjuk dan tangan, tunjuk ke arah yang sama dengan pointer yang sedang berjalan, dan dapatkan skor.

Sebuah pointer akan muncul setiap 3 detik dan akan berjalan naik menuju target, gerakkan tangan sesuai dengan pointer yang aktif untuk mendapatkan skor. Game akan berhenti ketika 

## Running Game
    pip install -r requirements.txt
    python main.py

First run downloads a small (~8 MB) MediaPipe hand model automatically (needs internet once).
Python 3.9 - 3.12 recommended.
use --camera 1 for another webcam, --no-music to mute

## Hand signs (tampil secara miror)
| Lane        | Sign                                    |
|-------------|-----------------------------------------|
| Up-left     | Telunjuk ke arah atas-kiri              |
| Up-right    | Telunjuk ke arah atas kanan             |
| Down-left   | Telunjuk ke arah kiri bawah             |
| Down-right  | Telunjuk ke arah kanan bawah            |
| Middle      | Menggenggam                             |
| Start/again | Buka tangan selama 1detik               |


## Tuning (top of main.py)
`DIAG_TOLERANCE`, `CONFIRM_FRAMES`, `EXTEND_RATIO` Untuk mengontrol deteksi,
`SPEED`, `SPAWN_EVERY`, `HIT_FROM_Y`, `MAX_MISSES` Untuk melakukan kontrol difficulty.



Submisi UTS Literasi Digital Kelas LD148
Divananda Ayu S | 24051204187 | Teknik Informatika
