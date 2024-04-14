#pragma once

#include <stdint.h>

#define MINST_MAX_RANK 4

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * @brief Enumerates the possibles errors that may occur when reading a MINST
   * file.
   * */
  enum minst_error
  {
    /**
     * @brief Indicates no error has occurred.
     * */
    MINST_ERR_NONE,
    /**
     * @brief A memory allocation failure occurred.
     * */
    MINST_ERR_OUT_OF_MEMORY,
    /**
     * @brief Indicates an error occurred in the callback function.
     * */
    MINST_ERR_CALLBACK,
    /**
     * @brief Indicates an error occurred while opening the samples file.
     * */
    MINST_ERR_OPEN_SAMPLES,
    /**
     * @brief Indicates an error occurred while opening the labels file.
     * */
    MINST_ERR_OPEN_LABELS,
    /**
     * @brief The shape is not what the caller expects.
     * */
    MINST_ERR_SHAPE,
    /**
     * @brief The type is not what the caller expects.
     * */
    MINST_ERR_TYPE,
    /**
     * @brief The file is missing data.
     * */
    MINST_ERR_MISSING_DATA,
    /**
     * @brief The type listed on the file was not recognized.
     * */
    MINST_ERR_UNKNOWN_TYPE,
    /**
     * @brief The sampler callback function failed.
     * */
    MINST_ERR_SAMPLER,
    /**
     * @brief Failed to go to a specific file location.
     * */
    MINST_ERR_SEEK
  };

  /**
   * @brief Enumerates the known MINST types.
   * */
  enum minst_type
  {
    MINST_TYPE_U8,
    MINST_TYPE_I8,
    MINST_TYPE_I16,
    MINST_TYPE_I32,
    MINST_TYPE_F32,
    MINST_TYPE_F64
  };

  /**
   * @brief Used for specifying the expected format of a MINST file.
   * */
  struct minst_format
  {
    /**
     * @brief The expected data type.
     * */
    enum minst_type type;

    /**
     * @brief The expected tensor rank (0 is scalar, 1 is vector, 2 is a matrix,
     *        etc).
     * */
    uint8_t rank;

    /**
     * @brief The shape of the tensor. All unused values should be set to 1. The first element is considered to be the
     *        number of tensors in the file.
     * */
    uint32_t shape[MINST_MAX_RANK];
  };

  /**
   * @brief A type definition for the function used to pass sample data to.
   *
   * @param callback_data User defined data passed from the calling environment.
   *
   * @param sample The sample data, in row major format.
   *
   * @param label The ground truth label for this sample.
   *
   * @return The predicted class index by the callback, or negative one if an
   * error occurred.
   * */
  typedef int (*minst_callback)(void* callback_data, const void* sample, const void* label);

  /**
   * @brief Chooses a sample from the dataset.
   *
   * @param callback_data The user defined sampler data passed from the calling environment.
   *
   * @param num_elements The number of elements in the dataset.
   *
   * @param element_idx The index of the chosen element.
   *
   * @return Zero on success, negative one on failure.
   * */
  typedef int (*minst_sampler)(void* sampler_data, uint32_t num_elements, uint32_t* element_idx);

  /**
   * @brief Converts an error enum to a human-readable string.
   *
   * @param err The error to convert to a string.
   *
   * @return The human-readable error string.
   * */
  const char* minst_strerror(enum minst_error err);

  /**
   * @brief Computes the size, in bytes, of one element in a given format.
   *
   * @param fmt The format to compute the element size of.
   *
   * @return The number of bytes that one element consists of.
   * */
  uint32_t minst_element_size(const struct minst_format* fmt);

  /**
   * @brief Loops through the dataset.
   *
   * @param samples_path The path to the samples file.
   *
   * @param labels_path The path to the labels file.
   *
   * @param callback_data Optional user-defined data to pass to the callback
   *                      functions.
   *
   * @param callback The function to pass the sample and label data to.
   *
   * @param sampler An optional user-defined sampler (the default is random
   *                sampling via Fisher–Yates shuffle).
   *
   * @return If an error occurs, it is returned by this function. Otherwise, @ref MINST_ERR_NONE is returned.
   * */
  enum minst_error minst_eval(const char* samples_path,
                              const char* labels_path,
                              const struct minst_format* sample_format,
                              const struct minst_format* label_format,
                              uint32_t batch_size,
                              void* callback_data,
                              const minst_callback callback,
                              void* sampler_data,
                              minst_sampler sampler);

  extern const struct minst_format minst_fashion_train_sample_format;

  extern const struct minst_format minst_fashion_train_label_format;

  extern const struct minst_format minst_fashion_test_sample_format;

  extern const struct minst_format minst_fashion_test_label_format;

#ifdef __cplusplus
} /* extern "C" */
#endif
