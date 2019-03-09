# CUI

```bash
$ docker-compose up -d --build
$ docker-compose exec python3 bash
```

* play with random

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py play-random
```

* play with primitive monte carlo

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py play-primitive-monte-carlo
```

* play with monte carlo tree search

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py play-mcts
```

* play with supervised learning policy network with initial state dual network

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py play-sl-policy-network-random
```

* play with supervised learning policy network with dual network

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py play-sl-policy-network data/model_xxxxxxxx.dat
```

* play with asynchronous policy and value monte carlo tree search with initial state dual network

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py play-apv-mcts-random
```

* play with asynchronous policy and value monte carlo tree search with dual network

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py play-sl-policy-network data/model_xxxxxxxx.dat
```

* replay play data

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py replay data/playdata_xxxxxxxx.dat
```

* create a trained dual network

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py create-model
```

* create a trained dual network to s3 bucket  
The procedure for building the AWS Batch environment is in `setup_batch.sh`.

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py create-model-batch "test-batch-bucket-name"
```

* training a dual network

```bash
$ export PYTHONIOENCODING=utf-8; python3 main.py create-model data/model_xxxxxxxx.dat
```

```bash
$ docker-compose down
```

* create a trained dual network with AWS Batch  
See [Setup Batch Environment](docs/setup_batch.md).

# WebUI

```bash
docker build -f Dockerfile_app -t heta-reversi-app:latest .
docker run -d -p 5000:5000 heta-reversi-app
```

http://localhost:5000/

```bash
docker stop [CONTAINER ID]
```

* create a ECS environment  
See [Setup APP Environment](docs/setup_app.md).
