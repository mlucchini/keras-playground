# Keras playground

Playground for keras neural networks

## Run

```
pip install -r requirements.txt
python networks/imdb-sentiment-word-embedding-network.py
```

## TensorBoard

```
tensorboard --logdir=logs
open http://localhost:6006/
```

## Install OpenCV

```
brew tap homebrew/science
brew install opencv
opencv_version=$(brew list opencv --versions |cut -d " " -f 2)
ln -s /usr/local/Cellar/opencv/${opencv_version}/lib/python2.7/site-packages/cv.py cv.py
ln -s /usr/local/Cellar/opencv/${opencv_version}/lib/python2.7/site-packages/cv2.so cv2.so
```

## Install Spark and Elephas

```
brew install apache-spark
spark_version=$(brew list apache-spark --versions |cut -d " " -f 2)

export SPARK_HOME=/usr/local/Cellar/apache-spark/${spark_version}/libexec
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.4-src.zip:$PYTHONPATH

# pyspark should run properly
pyspark

# Elephas 0.3 is not compatible with keras 1.2.0
pip install --user git+ssh://git@github.com/maxpumperla/elephas.git
```

## Ad-hoc

```
token=<slack-api-token>
pmchannel=<slack-api-pm-channel-id>
mpmchannel=<slack-api-pm-channel-id>
user=<slack-api-user-id>
```

```
curl https://slack.com/api/im.history?token=$token&channel=$pmchannel&pretty=1&count=1000 > slack-pm-$pmchannel.json
curl https://slack.com/api/mpim.history?token=$token&channel=$channel&pretty=1&count=1000 > slack-mpm-$mpmchannel.json

cat slack-pm-$pmchannel.json | jq '.messages[] | select(.user == "'$user'").text' -r > slack-$user.txt
cat slack-mpm-$mpmchannel.json | jq '.messages[] | select(.user == "'$user'").text' -r >> slack-$user.txt
```
