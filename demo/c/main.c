#include "minst.h"

#include <stdio.h>
#include <stdlib.h>

static int
callback(void* callback_data, const void* sample_data, const void* label_data)
{
  return 0;
}

int
main()
{
  const int batch_count = 1;

  const enum minst_error result = minst_eval("train-images-idx3-ubyte",
                                             "train-labels-idx1-ubyte",
                                             &minst_fashion_train_sample_format,
                                             &minst_fashion_train_label_format,
                                             batch_count,
                                             NULL,
                                             callback,
                                             NULL,
                                             NULL);
  if (result == MINST_ERR_NONE) {
    printf("success\n");
  } else {
    printf("failure: %s\n", minst_strerror((enum minst_error)result));
    return EXIT_FAILURE;
  }

  return 0;
}
