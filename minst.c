#include "minst.h"

#include <stdio.h>
#include <stdlib.h>

const struct minst_format minst_fashion_train_sample_format = { MINST_TYPE_U8, 3, { 60000, 28, 28, 1 } };

const struct minst_format minst_fashion_train_label_format = { MINST_TYPE_U8, 1, { 60000, 1, 1, 1 } };

const struct minst_format minst_fashion_test_sample_format = { MINST_TYPE_U8, 3, { 10000, 28, 28, 1 } };

const struct minst_format minst_fashion_test_label_format = { MINST_TYPE_U8, 1, { 10000, 1, 1, 1 } };

const char*
minst_strerror(const enum minst_error err)
{
  switch (err) {
    case MINST_ERR_NONE:
      return "no error";
    case MINST_ERR_OUT_OF_MEMORY:
      return "out of memory";
    case MINST_ERR_CALLBACK:
      return "callback function error";
    case MINST_ERR_OPEN_SAMPLES:
      return "failed to open samples file";
    case MINST_ERR_OPEN_LABELS:
      return "failed to open labels file";
    case MINST_ERR_SHAPE:
      return "unexpected shape";
    case MINST_ERR_TYPE:
      return "unexpected type";
    case MINST_ERR_MISSING_DATA:
      return "missing data";
    case MINST_ERR_UNKNOWN_TYPE:
      return "unknown type";
    case MINST_ERR_SAMPLER:
      return "sampler function error";
    case MINST_ERR_SEEK:
      return "failed to seek file location";
  }

  return "unknown error";
}

struct magic
{
  enum minst_type type;

  uint8_t rank;
};

static enum minst_error
minst_read_magic(FILE* file, struct magic* m)
{
  uint8_t data[4];
  unsigned long read_size;

  read_size = fread(data, sizeof(data), 1, file);
  if (read_size != 1) {
    return MINST_ERR_MISSING_DATA;
  }

  switch (data[2]) {
    case 0x08:
      m->type = MINST_TYPE_U8;
      break;
    case 0x09:
      m->type = MINST_TYPE_I8;
      break;
    case 0x0B:
      m->type = MINST_TYPE_I16;
      break;
    case 0x0C:
      m->type = MINST_TYPE_I32;
      break;
    case 0x0D:
      m->type = MINST_TYPE_F32;
      break;
    case 0x0E:
      m->type = MINST_TYPE_F64;
      break;
    default:
      return MINST_ERR_UNKNOWN_TYPE;
  }

  m->rank = data[3];

  return MINST_ERR_NONE;
}

static enum minst_error
minst_check_format(FILE* file, const struct minst_format* format)
{
  struct magic m;
  enum minst_error err;
  uint32_t dim_size;
  uint32_t dim_idx;
  uint8_t read_buf[4];

  err = minst_read_magic(file, &m);
  if (err != MINST_ERR_NONE) {
    return err;
  }

  if (m.type != format->type) {
    return MINST_ERR_TYPE;
  }

  if (m.rank != format->rank) {
    printf("HERE (line=%d)\n", __LINE__);
    return MINST_ERR_SHAPE;
  }

  for (dim_idx = 0; dim_idx < m.rank; dim_idx++) {

    if (fread(&read_buf[0], sizeof(read_buf), 1, file) != 1) {
      return MINST_ERR_TYPE;
    }

    dim_size = 0;

    dim_size |= ((uint32_t)read_buf[0]) << 24u;
    dim_size |= ((uint32_t)read_buf[1]) << 16u;
    dim_size |= ((uint32_t)read_buf[2]) << 8u;
    dim_size |= ((uint32_t)read_buf[3]);

    if (format->shape[dim_idx] != dim_size) {
      printf("HERE (line=%d) (shape=%d)\n", __LINE__, dim_size);
      return MINST_ERR_SHAPE;
    }
  }

  return MINST_ERR_NONE;
}

static uint32_t
minst_type_size(enum minst_type type)
{
  switch (type) {
    case MINST_TYPE_U8:
    case MINST_TYPE_I8:
      return 1;
    case MINST_TYPE_I16:
      return 2;
    case MINST_TYPE_F32:
    case MINST_TYPE_I32:
      return 4;
    case MINST_TYPE_F64:
      return 8;
  }
  return 0;
}

uint32_t
minst_element_size(const struct minst_format* fmt)
{
  return fmt->shape[1] * fmt->shape[2] * fmt->shape[3] * minst_type_size(fmt->type);
}

static long int
minst_element_offset(const struct minst_format* format, const uint32_t element_idx)
{
  long int offset;

  offset = 4 /* magic number size */;

  offset += ((long int)format->rank) * ((long int)sizeof(uint32_t)) /* shape description */;

  offset +=
    ((long int)element_idx) * format->shape[1] * format->shape[2] * format->shape[3] * minst_type_size(format->type);

  return offset;
}

struct default_sampler
{
  uint32_t* indices;

  uint32_t idx;
};

static uint32_t
minst_rand(uint32_t min_v, uint32_t max_v)
{
  return (((uint32_t)rand()) % (max_v - min_v)) + min_v;
}

static int
minst_default_sampler(void* sampler_data, const uint32_t num_elements, uint32_t* element_idx)
{
  struct default_sampler* data;
  uint32_t i;
  uint32_t j;
  uint32_t tmp;

  data = sampler_data;

  if (!data->indices) {

    data->indices = malloc(num_elements * sizeof(uint32_t));
    if (data->indices == NULL) {
      return -1;
    }

    for (i = 0; i < num_elements; i++) {
      data->indices[i] = i;
    }

    for (i = 1; i < num_elements; i++) {
      j = minst_rand(0, i);
      tmp = data->indices[i];
      data->indices[i] = data->indices[j];
      data->indices[j] = tmp;
    }
  }

  *element_idx = data->indices[data->idx];

  return 0;
}

static enum minst_error
minst_eval_impl(FILE* samples_file,
                FILE* labels_file,
                const struct minst_format* sample_format,
                const struct minst_format* label_format,
                const uint32_t batch_size,
                void* callback_data,
                const minst_callback callback,
                void* sampler_data,
                const minst_sampler sampler)
{
  enum minst_error error;
  uint32_t num_samples;
  uint32_t iteration;
  uint8_t* sample_buffer;
  uint8_t* label_buffer;
  uint32_t batch_idx;
  uint32_t element_idx;
  long int sample_offset;
  long int label_offset;
  uint32_t sample_size;
  uint32_t label_size;

  error = minst_check_format(samples_file, sample_format);
  if (error != MINST_ERR_NONE) {
    return error;
  }

  error = minst_check_format(labels_file, label_format);
  if (error != MINST_ERR_NONE) {
    return error;
  }

  num_samples = sample_format->shape[0];

  sample_size = minst_element_size(sample_format);

  label_size = minst_element_size(label_format);

  sample_buffer = malloc(batch_size * sample_size);
  if (sample_buffer == NULL) {
    return MINST_ERR_OUT_OF_MEMORY;
  }

  label_buffer = malloc(batch_size * label_size);
  if (label_buffer == NULL) {
    free(sample_buffer);
    return MINST_ERR_OUT_OF_MEMORY;
  }

  for (iteration = 0; iteration < num_samples; iteration += batch_size) {

    for (batch_idx = 0; batch_idx < batch_size; batch_idx++) {

      if (sampler(sampler_data, num_samples, &element_idx) != 0) {
        free(sample_buffer);
        free(label_buffer);
        return MINST_ERR_SAMPLER;
      }

      sample_offset = minst_element_offset(sample_format, element_idx);

      if (fseek(samples_file, sample_offset, SEEK_SET) != 0) {
        free(sample_buffer);
        free(label_buffer);
        return MINST_ERR_SEEK;
      }

      label_offset = minst_element_offset(label_format, element_idx);

      if (fseek(labels_file, label_offset, SEEK_SET) != 0) {
        free(sample_buffer);
        free(label_buffer);
        return MINST_ERR_SEEK;
      }

      if (fread(sample_buffer + sample_size * batch_idx, sample_size, 1, samples_file) != 1) {
        free(sample_buffer);
        free(label_buffer);
        return MINST_ERR_MISSING_DATA;
      }

      if (fread(label_buffer + label_size * batch_idx, label_size, 1, labels_file) != 1) {
        free(sample_buffer);
        free(label_buffer);
        return MINST_ERR_MISSING_DATA;
      }
    }

    if (callback(callback_data, sample_buffer, label_buffer) != 0) {
      free(sample_buffer);
      free(label_buffer);
      return MINST_ERR_CALLBACK;
    }
  }

  free(sample_buffer);
  free(label_buffer);
  return MINST_ERR_NONE;
}

enum minst_error
minst_eval(const char* samples_path,
           const char* labels_path,
           const struct minst_format* sample_format,
           const struct minst_format* label_format,
           const uint32_t batch_size,
           void* callback_data,
           const minst_callback callback,
           void* sampler_data,
           minst_sampler sampler)
{
  FILE* samples_file;
  FILE* labels_file;
  enum minst_error err;
  struct default_sampler def_sampler;

  def_sampler.idx = 0;
  def_sampler.indices = NULL;

  if (!sampler) {
    sampler_data = &def_sampler;
    sampler = minst_default_sampler;
  }

  samples_file = fopen(samples_path, "rb");
  if (samples_file == NULL) {
    return MINST_ERR_OPEN_SAMPLES;
  }

  labels_file = fopen(labels_path, "rb");
  if (labels_file == NULL) {
    fclose(samples_file);
    return MINST_ERR_OPEN_LABELS;
  }

  err = minst_eval_impl(
    samples_file, labels_file, sample_format, label_format, batch_size, callback_data, callback, sampler_data, sampler);

  /* cleanup */

  free(def_sampler.indices);

  fclose(labels_file);

  fclose(samples_file);

  return err;
}
