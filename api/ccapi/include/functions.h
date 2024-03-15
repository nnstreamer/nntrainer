// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   functions.h
 * @date   15 March 2024
 * @brief  NNTrainer API for IR graph configurations.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug	   No known bugs except for NYI items
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_FUNCTIONS_H__
#define __ML_TRAIN_FUNCTIONS_H__

#if __cplusplus < MIN_CPP_VERSION
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus

#include <iostream>
#include <memory>
#include <vector>

namespace ml {
namespace train {

    class Function;

    /**
     * @brief TensorCore for graph configuration.
     */
    class TensorNode {
    public:
        bool is_leaf = true;
        bool requires_grad = false;
        std::shared_ptr<Function> creator = nullptr;

        /**
         * @brief Construct a new TensorNode
         */
        TensorNode(): is_leaf(true), requires_grad(false) {}

        /**
         * @brief Construct a new TensorNode from Function
         */
        TensorNode(std::shared_ptr<Function> &func):
            is_leaf(false), requires_grad(true), creator(func) {}
    };

    /**
     * @brief Tensor API for users
     */
    class Tensor {
    private:
        std::shared_ptr<TensorNode> node = nullptr;
    public:
        /**
         * @brief Construct a new Tensor
         */
        Tensor() {
            node = std::make_shared<TensorNode>();
        }

        /**
         * @brief Construct a new Tensor from Function
         */
        Tensor(std::shared_ptr<Function> &func) {
            node = std::make_shared<TensorNode>(func);
        }

        /**
         * @brief Construct a new Tensor using TensorNode
         */
        Tensor(std::shared_ptr<TensorNode> &node) {
            node = node;
        }

        /**
         * @brief Check if the tensor is a leaf tensor
         */
        bool is_leaf() {
            return node->is_leaf;            
        }

        /**
         * @brief Check if the tensor requires gradients
         */
        bool get_requires_grad() {            
            return node->requires_grad;
        }

        /**
         * @brief Set the requires_grad flag of the tensor
         */
        bool set_requires_grad(bool requires_grad) {
            return node->requires_grad = requires_grad;
        }

        /**
         * @brief Return the creator function of the tensor
         */
        std::shared_ptr<Function> get_creator() {
            return node->creator;
        }

        /**
         * @brief Return the tensor node
         */
        std::shared_ptr<TensorNode> get_node() {
            return node;
        }
    };

    /**
     * @brief Function(operation) API
     */
    class Function {
    private:
        /**
         * @brief Keep input tensors of the function
         */
        std::vector<std::shared_ptr<TensorNode>> inputs;

        /**
         * @brief Keep output tensors of the function
         */
        std::vector<std::shared_ptr<TensorNode>> outputs;
    public:
        std::string op_type = "";

        /**
         * @brief Forwarding operation for graph configurations
         */
        std::vector<Tensor> forward(std::shared_ptr<Function> &func,
                                    std::vector<Tensor> xs, int num_output_tensors=1) {
            std::vector<Tensor> ys = std::vector<Tensor>();
            
            for (int i=0; i < (int)xs.size(); ++i) {
                inputs.push_back(xs[i].get_node());
            }            
            
            for (int i=0; i < num_output_tensors; ++i) {
                Tensor t = Tensor(func);
                outputs.push_back(t.get_node());
                ys.push_back(t);
            }
            
            return ys;
        }

        /**
         * @brief Get input tensors of the function
         */
        std::vector<std::shared_ptr<TensorNode>> get_inputs() {
            return inputs;
        }

        /**
         * @brief Get output tensors of the function
         */
        std::vector<std::shared_ptr<TensorNode>> get_outputs() {
            return outputs;
        }
    };

    /**
     * @brief Add Function Class
     */
    class Add : public Function {
    public:
        Add(): Function() { op_type = "add"; }
    };

    /**
     * @brief Sub Function Class
     */
    class Sub : public Function {
    public:
        Sub(): Function() { op_type = "sub"; }
    };

    /**
     * @brief Mul Function Class
     */
    class Mul : public Function {
    public:
        Mul(): Function() { op_type = "mul"; }
    };

    /**
     * @brief Div Function Class
     */
    class Div : public Function {
    public:
        Div(): Function() { op_type = "div"; }
    };

    /**
     * @brief Pow Function Class
     */
    class Pow : public Function {
    public:
        Pow(): Function() { op_type = "pow"; }
    };

    /**
     * @brief Add operation api
     */
    Tensor add(Tensor &x1, Tensor &x2) {
        std::shared_ptr<Function> f = std::make_shared<Add>();
        return f->forward(f, {x1, x2})[0];
    };

    /**
     * @brief Sub operation api
     */
    Tensor sub(Tensor &x1, Tensor &x2) {
        std::shared_ptr<Function> f = std::make_shared<Sub>();
        return f->forward(f, {x1, x2})[0];
    };

    /**
     * @brief Mul operation api
     */
    Tensor mul(Tensor &x1, Tensor &x2) {
        std::shared_ptr<Function> f = std::make_shared<Mul>();
        return f->forward(f, {x1, x2})[0];
    };

    /**
     * @brief Div operation api
     */
    Tensor div(Tensor &x1, Tensor &x2) {
        std::shared_ptr<Function> f = std::make_shared<Div>();
        return f->forward(f, {x1, x2})[0];
    };

    /**
     * @brief Pow operation api
     */
    Tensor pow(Tensor &x) {
        std::shared_ptr<Function> f = std::make_shared<Pow>();
        return f->forward(f, {x})[0];
    };

}
}

#endif // __ML_TRAIN_FUNCTIONS_H__
