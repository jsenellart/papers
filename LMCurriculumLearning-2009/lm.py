from collections import Counter
from pathlib import Path
import sys
import random
import math
from tqdm import tqdm

import configargparse

parser = configargparse.ArgParser(config_file_parser_class=configargparse.DefaultConfigFileParser)
parser.add('-c', '--config', is_config_file=True, help='config file path')
parser.add('-C', '--create_config', help='create config file')
parser.add("--train", help="training data", required=True)
parser.add("--test", help="testing data", required=True)
parser.add("--include_unk", "-u", action="store_true")
parser.add("--test_size", type=int, default=1000)
parser.add("--buffer_size", type=int, default=500000)
parser.add("--batch_size", type=int, default=512)
parser.add("--test_batch_size", type=int, default=32)
parser.add("--steps_per_epoch", type=int, default=4096)
parser.add("--vocab", "-v", type=int, default=20000)
parser.add("--vocab_file", help="vocabulary file", default=None)
parser.add("--verbose", type=int, default=1)
parser.add("--model", default="./lmmodel")
parser.add("--sgd_learning_rate", default=0.01, type=float)
parser.add("--sgd_decay", default=1e-6, type=float)
parser.add("--sgd_momentum", default=0.9, type=float)
parser.add("--curriculum_steps", default=1, type=int)
parser.add("--curriculum_examples", default=100000000, type=int)

args = parser.parse_args()

# Display all of values - and where they are coming from
print(parser.format_values())

if args.create_config:
  options = {}
  for attr, value in args.__dict__.items():
    if attr != "config" and attr != "create_config" and value is not None:
      options[attr] = value
  file_name = args.create_config
  content = configargparse.DefaultConfigFileParser().serialize(options)
  Path(file_name).write_text(content)
  print("configuration saved to file: %s" % file_name)
  sys.exit(0)

import tensorflow as tf
import tensorflow.keras as K
import tensorflow_datasets as tfds

# Implement simple SpaceTokenizer - built-in tokenizer in tf filter-out
# non alphanumeric tokens
# see https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/TokenTextEncoder
class SpaceTokenizer(object):
  def tokenize(self, s):
    toks = []
    toks.extend(tf.compat.as_text(s).split(' '))
    toks = [t for t in toks if t]
    return toks

tokenizer = SpaceTokenizer()

train_dataset = tf.data.TextLineDataset(args.train)
test_dataset = tf.data.TextLineDataset(args.test)

if not args.vocab_file:
  VOC_SIZE = args.vocab

  print("Read Corpus - prepare sorted vocabulary")
  freq = Counter()
  for text_tensor in train_dataset:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    sentence_vocabulary_set = set(some_tokens)
    for v in sentence_vocabulary_set:
      freq[v] += 1

  vocab = [k for (k,v) in freq.most_common()]
else:
  print("Read Vocab file - assuming it is sorted by decreasing frequency")
  vocab = []
  with open(args.vocab_file) as f:
    for l in f:
      vocab.append(l.strip())

VOC_SIZE = min(len(vocab), args.vocab)
print("Total Vocab Size=", len(vocab), "Actual vocab size=", VOC_SIZE)
vocab = vocab[:VOC_SIZE]

if args.include_unk:
  VOC_SIZE += 1

#let us define a tensor with all vocabs - that we will use in test
#to find rank of a given context-word score
vocabs = tf.cast(tf.constant(range(1,VOC_SIZE+1)),
                 dtype=tf.int64)

encoder = tfds.features.text.TokenTextEncoder(vocab, tokenizer=tokenizer)

def encode(text_tensor):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, True

def encode_map_fn(text):
  encoded_text, _ = tf.py_function(encode, 
                                   inp=[text], 
                                   Tout=(tf.int64, tf.bool))
  encoded_text.set_shape([None])
  return encoded_text

competence = tf.Variable(1.0)
nexamples = tf.Variable(0, dtype=tf.int64)

def window_train(token_list):
  windows = []
  labels = []
  for i in range(len(token_list)-4):
    max_window = tf.math.reduce_max(token_list[i:i+5])
    if max_window <= tf.cast(competence*VOC_SIZE, tf.int64):
      windows.append(token_list[i:i+5])
      labels.append(0)
      fake=[token_list[i],
            token_list[i+1],
            token_list[i+2],
            token_list[i+3],
            tf.random.uniform([],
                              minval=1,
                              maxval=tf.cast(competence*VOC_SIZE,tf.dtypes.int64),
                              dtype=tf.dtypes.int64)]
      windows.append(fake)
      labels.append(1)
  nexamples.assign_add(tf.cast(len(token_list)-4, tf.int64))
  return windows, labels

def window_map_train_fn(token_list):
  windows, labels = tf.py_function(window_train, 
                                   inp=[token_list], 
                                   Tout=(tf.int64, tf.float32))

  windows.set_shape([None,5])
  labels.set_shape([None])

  return windows, labels

def window_test(token_list):
  windows = []
  labels = []
  for i in range(len(token_list)-4):
    max_window = tf.math.reduce_max(token_list[i:i+5])
    if max_window <= tf.cast(VOC_SIZE, tf.int64):
      windows.append(token_list[i:i+5])
      labels.append(0)
  return windows, labels

def window_map_test_fn(token_list):
  windows, _ = tf.py_function(window_test, 
                                   inp=[token_list], 
                                   Tout=(tf.int64, tf.float32))

  windows.set_shape([None,5])

  return windows

def filter_nonempty_xy(ds): 
  return ds.filter(lambda x, y: len(x) > 0)

def filter_nonempty_x(ds): 
  return ds.filter(lambda x: len(x) > 0)

# Train Data preparation
encoded_train_data = train_dataset.map(encode_map_fn)
train_data = encoded_train_data.shuffle(args.buffer_size)
# apply window operator, and remove empty list
train_data = train_data.map(window_map_train_fn).apply(filter_nonempty_xy)
train_data = train_data.flat_map(lambda x,y: tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y))))
train_data = train_data.batch(args.batch_size*2)

# Test Data preparation
test_data = test_dataset.map(encode_map_fn)
# apply window operator, and remove empty list
test_data = test_data.map(window_map_test_fn).apply(filter_nonempty_x)
test_data = test_data.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
test_data = test_data.take(args.test_size)
test_data = test_data.batch(args.test_batch_size)

# Definition of the model and loss function
main_input = K.layers.Input(shape=(5), dtype='int32', name='main_input')
embedding = K.layers.Embedding(VOC_SIZE+1, 50)(main_input)
o1 = K.layers.Reshape((250,))(embedding)
o2 = K.layers.Dense(100, activation='tanh')(o1)
predictions = K.layers.Dense(1)(o2)

def loss_fnc(y_true, y_pred):
  positive = y_pred[0::2]
  negative = y_pred[1::2]
  loss = tf.maximum(0., 1. - positive + negative)
  loss = tf.reshape(tf.tile(tf.reshape(loss,[tf.size(loss),1]),[1,2]),[2*tf.size(loss)])
  return loss

model = K.Model(inputs=main_input, outputs=predictions)

print(model.summary())

# Train the model
sgd = K.optimizers.SGD(lr=args.sgd_learning_rate,
                       decay=args.sgd_decay,
                       momentum=args.sgd_momentum,
                       nesterov=True)
model.compile(sgd, loss=loss_fnc)

ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                           optimizer=sgd,
                           net=model,
                           nexamples=nexamples)
manager = tf.train.CheckpointManager(ckpt, args.model, max_to_keep=3)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
  print("Total number of examples=", nexamples.numpy())
else:
  print("Initializing from scratch.")

summary_writer = tf.summary.create_file_writer(args.model)

while True:
  rate = (int(nexamples.numpy()*1.0/args.curriculum_examples)+1.0)/args.curriculum_steps
  competence.assign(tf.cast(tf.math.minimum(rate, 1.0), tf.float32))

  # Train
  if ckpt.save_counter != 0:
    h = model.fit(train_data, steps_per_epoch=args.steps_per_epoch, verbose=args.verbose)
    loss = h.history["loss"][0]
    ckpt.step.assign_add(args.steps_per_epoch)
  else:
    loss = None
    first = False

  # Eval
  test_name = "%s/test_%d.out" % (args.model, ckpt.save_counter)
  print("Evaluating model => ", test_name)
  with open(test_name, "w") as ftest:
    sum_logrank = 0
    for test in tqdm(test_data,
                     unit="batch",
                     ncols=80,
                     total=math.ceil(args.test_size*1.0/args.test_batch_size)):
      t_expanded = tf.concat([tf.reshape(tf.tile(test[:,:4],[1,VOC_SIZE]),
                                         [-1, 4]),
                              tf.reshape(tf.tile(vocabs,[test.shape[0]]),
                                         [-1, 1])],
                             axis=1)
      out = model.predict(t_expanded)
      out = tf.reshape(out, [-1, VOC_SIZE])
      for ib in range(test.shape[0]):
        w = test[ib][4]-1
        windows_s = "["+" ".join([vocab[t-1] for t in test[ib][:-1]])+ "]..."+vocab[w]
        out_w = out[ib][w]
        sorted_out = tf.sort(tf.reshape(out[ib],[VOC_SIZE]), direction='DESCENDING')
        rank = tf.where(tf.equal(sorted_out,out_w))[:,0][0]+1
        best10 =" ".join([vocab[tf.where(tf.equal(out[ib], sorted_out[idx]))[:,0][0].numpy()]+"/"+
                            str(sorted_out[idx].numpy())
                          for idx in range(10)])
        ftest.write("\t".join((windows_s, str(rank.numpy()),
                         str(tf.math.log(tf.cast(rank,dtype=tf.float32)).numpy()), best10))+"\n")
        sum_logrank += tf.math.log(tf.cast(rank,dtype=tf.float32))
    ftest.write("======\n%f\n" % (sum_logrank.numpy()/args.test_size))

  print(ckpt.step.numpy()*args.batch_size, "==>", "total windows", nexamples.numpy(), "competence", competence.numpy(), "loss", loss, "logrank", sum_logrank.numpy()/args.test_size)
  with summary_writer.as_default():
    tf.summary.scalar('logrank', sum_logrank/args.test_size, step=ckpt.step*args.batch_size)
    tf.summary.scalar('competence', competence, step=ckpt.step*args.batch_size)
    tf.summary.scalar('total windows', nexamples, step=ckpt.step*args.batch_size)
    if loss is not None:
      tf.summary.scalar('train_loss', loss, step=ckpt.step*args.batch_size)
  manager.save()
