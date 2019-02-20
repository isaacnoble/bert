# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import time

import modeling
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


class InputExample(object):
  def __init__(self, unique_id, is_context, tokens):
    self.unique_id = unique_id
    self.is_context = is_context
    self.tokens = tokens


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, is_context, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.is_context = is_context
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_is_context = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_is_context.append(feature.is_context)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "is_contexts":
            tf.constant(all_is_context, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, use_tpu, use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    is_contexts = features["is_contexts"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    predictions = {
        "unique_id": unique_ids,
        "is_context": is_contexts,
        "embeddings": model.get_embedding_output(),
        "transformer": model.get_sequence_output()
    }

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in example.tokens:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    if ex_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (example.unique_id))
      tf.logging.info("is_context: %s" % (example.is_context))
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

    features.append(
        InputFeatures(
            unique_id=example.unique_id,
            is_context=example.is_context,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids))
  return features


def _truncate_seq_pair(tokens, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens)
    if total_length <= max_length:
      break
    tokens.pop()


def read_documents(input_files, tokenizer):
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)
  return all_documents
  
def build_examples(documents, max_seq_length):
  examples = []
  unique_id = 0
  for document in documents:
    for context_end in range(1, len(document)-1):
      num_tokens = 0
      context_start = context_end
      # Account for [CLS] and [SEP] with "- 2"
      while context_start > 0 and num_tokens + len(document[context_start - 1]) < max_seq_length - 2:
        context_start -= 1
        num_tokens += len(document[context_start])

      context_tokens = [token for line in document[context_start:context_end] for token in line]
      
      if len(context_tokens) > max_seq_length - 2:
        raise ValueError('context tokens ({}) exceed max seq length ({}) with document range {}-{}'.format(len(context_tokens), max_seq_length - 2, context_start, context_end))
      
      num_tokens = 0
      real_end = context_end
      while real_end < len(document)-1 and num_tokens + len(document[real_end + 1]) < max_seq_length - 2:
        real_end += 1
        num_tokens += len(document[real_end])

      real_tokens = [token for line in document[context_end:real_end] for token in line]
      if len(real_tokens) > max_seq_length - 2:
        raise ValueError('real tokens ({}) exceed max seq length ({}) with document range {}-{}'.format(len(real_tokens), max_seq_length - 2, context_end, real_end))
      examples.append(
        InputExample(unique_id=unique_id, is_context=1, tokens=context_tokens))
      examples.append(
        InputExample(unique_id=unique_id, is_context=0, tokens=real_tokens))
      unique_id += 1
  return examples

def write_instance_to_example_files(generator_fn, output_files, hidden_size, splits=100):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    for split in range(splits):
        writers.append(tf.python_io.TFRecordWriter(output_file + "-" + str(split)))

  writer_index = 0

  total_written = 0
  num_to_print = 1000
  start = time.time()
  for (context, real) in generator_fn():
    context_ids = context["input_ids"]
    context_mask = context["input_mask"]
    context_embeddings = context["embeddings"]
    context_transformer = context["transformer"]

    real_ids = real["input_ids"]
    real_mask = real["input_mask"]
    real_embeddings = real["embeddings"]
    real_transformer = real["transformer"]
    embedded_size = real_embeddings.shape[-1]
    transformed_size = real_transformer.shape[-1]

    assert len(context_ids) == FLAGS.max_seq_length
    assert len(context_mask) == FLAGS.max_seq_length

    context_embeddings = context_embeddings.flatten()
    context_transformer = context_transformer.flatten()
    assert len(context_embeddings) == FLAGS.max_seq_length * hidden_size
    assert len(context_transformer) == FLAGS.max_seq_length * hidden_size

    assert len(real_ids) == FLAGS.max_seq_length
    assert len(real_mask) == FLAGS.max_seq_length

    real_embeddings = real_embeddings.flatten()
    real_transformer = real_transformer.flatten()
    assert len(real_embeddings) == FLAGS.max_seq_length * hidden_size
    assert len(real_transformer) == FLAGS.max_seq_length * hidden_size

    features = collections.OrderedDict()
    features["context_ids"] = create_int_feature(context_ids)
    features["context_mask"] = create_int_feature(context_mask)
    features["context_embeddings"] = create_float_feature(context_embeddings)
    features["context_transformer"] = create_float_feature(context_transformer)
    features["real_ids"] = create_int_feature(real_ids)
    features["real_mask"] = create_int_feature(real_mask)
    features["real_embeddings"] = create_float_feature(real_embeddings)
    features["real_transformer"] = create_float_feature(real_transformer)
    features["embedded_size"] = create_int_feature([embedded_size])
    features["transformed_size"] = create_int_feature([transformed_size])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if total_written % num_to_print == 0:
        print("Wrote {} instances at {} sec/example".format(total_written, (time.time() - start) / num_to_print))
        start = time.time()
        break

  for writer in writers:
    try:
        writer.close()
    except:
        print("Error closing writer!")

  tf.logging.info("Wrote %d total instances", total_written)

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  documents = read_documents(input_files, tokenizer)
  tf.logging.info("*** Read in %i documents ***", len(documents))
  examples = build_examples(documents, FLAGS.max_seq_length)
  tf.logging.info("*** Generated %i examples ***", len(examples))

  features = convert_examples_to_features(
      examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

  context_features = {}
  real_features = {}
  for feature in features:
    if feature.is_context:
      context_features[feature.unique_id] = feature
    else:
      real_features[feature.unique_id] = feature

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.batch_size)

  input_fn = input_fn_builder(
      features=features, seq_length=FLAGS.max_seq_length)

  def generator_fn():
    finished_samples = {}
    for result in estimator.predict(input_fn, yield_single_examples=True):
      unique_id = int(result["unique_id"])
      if unique_id in finished_samples:
        if int(result["is_context"]) == 1:
          yield (result, finished_samples[unique_id])
        else:
          yield (finished_samples[unique_id], result)
      else:
        finished_samples[unique_id] = result

  write_instance_to_example_files(generator_fn, [FLAGS.output_file], bert_config.hidden_size)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_file")
  tf.app.run()