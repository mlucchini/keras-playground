from elephas import optimizers as elephas_optimizers
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from pyspark import SparkContext, SparkConf


class TrainerConf:
    def __init__(self, model, epochs, batch_size, train_test_split, optimizer='adam', metrics=['accuracy'], verbose=1,
                 workers=1, master_url='local[8]'):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.optimizer = optimizer
        self.metrics = metrics
        self.verbose = verbose
        self.workers = workers
        self.master_url = master_url


class DistributedTrainer:
    def __init__(self, conf):
        self.model = conf.model
        self.epochs = conf.epochs
        self.batch_size = conf.batch_size
        self.train_test_split = conf.train_test_split
        self.verbose = conf.verbose
        self.conf = SparkConf().setAppName('App').setMaster(conf.master_url)
        self.sc = SparkContext(conf=self.conf)
        self.spark_model = SparkModel(self.sc, conf.model, optimizer=elephas_optimizers.get(conf.optimizer),
                                      frequency='epoch', master_metrics=conf.metrics, mode='asynchronous',
                                      num_workers=conf.workers)

    def fit(self, x_train, y_train):
        rdd = to_simple_rdd(self.sc, x_train, y_train)
        self.spark_model.train(rdd, nb_epoch=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                               validation_split=self.train_test_split)


class LocalTrainer:
    def __init__(self, conf):
        self.model = conf.model
        self.epochs = conf.epochs
        self.batch_size = conf.batch_size
        self.train_test_split = conf.train_test_split
        self.verbose = conf.verbose

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train, nb_epoch=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                       validation_split=self.train_test_split)
