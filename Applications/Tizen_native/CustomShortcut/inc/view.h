/**
 * @file view.h
 * @date 15 May 2020
 * @brief TIZEN Native Example App view entry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __nntrainer_example_custom_shortcut_view_H__
#define __nntrainer_example_custom_shortcut_view_H__

#include "data.h"
#include <Elementary.h>
#include <dlog.h>
#include <efl_extension.h>

/**
 * @brief initiate window and conformant.
 * @param[in] context context of widget instance
 * @param[in] w width
 * @param[in] h height
 * @retval #WIDGET_ERROR_*
 */
int view_init(widget_context_h context, int w, int h);

/**
 * @brief creates layout from edj
 * @param[in] context context of widget instance
 */
int view_create(widget_context_h context);

#endif /* __nntrainer_example_custom_shortcut_view_H__ */
