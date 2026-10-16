#if defined(NNTR_LFM2_RECENCY_TRACE) && NNTR_LFM2_RECENCY_TRACE
      if (from == 0 && input_.batch() == 1 && input_.channel() == 1 &&
          (total_tokens == 1024 || total_tokens == 2048) &&
          hidden_size == 2048 && experts_virtual && in_norm && !router_norm &&
          !out_norm && !out_add && context.getNumInputs() == 1 &&
          !softmax_router && !gelu_act && num_experts == 64 && topk == 4 &&
          std::get<nntrainer::props::Unit>(moe_props).get() == 1536)
        std::fprintf(
          stderr, "[LFM-RECENCY] layer=%u M=%u raw=%u kept=%zu fast=%u !\n",
          trace_layer, total_tokens, total_tokens * (topk + EXTRA_TOPK),
          extra_top_k.size(), compact_recency ? 1u : 0u);
#endif
