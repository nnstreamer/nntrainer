// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Donghoon Kang <dhkang01@snu.ac.kr>
 *
 * @file   channel_shuffle.h
 * @date   23 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghoon Kang <dhkang01@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 * @brief  This is Channel Shuffle Layer Class for Neural Network
 *
 */

#ifndef __CHANNEL_SHUFFLE_H_
#define __CHANNEL_SHUFFLE_H_
#ifdef __cplusplus

#include <memory.h>

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   ChannelShuffle
 * @brief   Channel Shuffle Layer
 */
class ChannelShuffle : public virtual Layer {
public:
    /**
     * @brief     Constructor of Channel Shuffle Layer
     */
    ChannelShuffle();

    /**
     * @brief     Destructor of Channel Shuffle Layer
     */
    ~ChannelShuffle() = default;

    /**
     * @brief  Move constructor of Channel Shuffle Layer.
     * @param[in] ChannelShuffle &&
     */
    ChannelShuffle(ChannelShuffle &&rhs) noexcept = default;

    /**
     * @brief  Move assignment operator.
     * @parma[in] rhs ChannelShuffle to be moved.
     */
    ChannelShuffle &operator=(ChannelShuffle &&rhs) = default;

    /**
     * @copydoc Layer::finalize(InitLayerContext &context)
     */
    void finalize(InitLayerContext &context) override;

    /**
     * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
     */
    void forwarding(RunLayerContext &context, bool training) override;

    /**
     * @copydoc Layer::calcDerivative(RunLayerContext &context)
     */
    void calcDerivative(RunLayerContext &context) override;

    /**
     * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods method)
     */
    void exportTo(Exporter &exporter,
                  const ml::train::ExportMethods &method) const override;

    /**
     * @copydoc Layer::getType()
     */
    const std::string getType() const override { return ChannelShuffle::type; };

    /**
     * @copydoc Layer::supportBackwarding()
     */
    bool supportBackwarding() const override { return true; };

    /**
     * @copydoc Layer::setProperty(const std::vector<std::string> &values)
     */
    void setProperty(const std::vector<std::string> &values) override;

    static constexpr const char *type = "channelshuffle";

private:
    int num_groups; /**< number of groups for channel shuffling */
};
}

#endif /* __cplusplus */
#endif /* __CHANNEL_SHUFFLE_H__ */