import sqlite3, sys
path = sys.argv[1] if len(sys.argv) > 1 else "phase_04_live.db"
c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
for (t,) in c.execute("select name from sqlite_master where type='table' order by name"):
    print(f"--- {t} ({c.execute(f'select count(*) from {t}').fetchone()[0]} rows)")
    for row in c.execute(f"select * from {t} order by rowid desc limit 5"):
        print("   ", row)
