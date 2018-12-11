```bash
$ docker-compose up -d --build
$ docker-compose exec python3 bash
```

* play

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py play
```

* replay play data

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py replay data/playdata_xxxxxxxx.csv
```

```bash
$ docker-compose down
```
