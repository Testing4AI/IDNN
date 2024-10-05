import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import random
from tensorflow import keras


os.environ["CUDA_VISIBLE_DEVICES"]='0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


model = keras.models.load_model("./tmp.h5")


class IDNN:
    def __init__(self, model, x_test, y_test):
        self.initial_model = model
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    
    def acc(self):
        _loss, _acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return _acc

    
    def initial_acc(self):
        _loss, _acc = self.initial_model.evaluate(self.x_test, self.y_test, verbose=0)
        return _acc
        

    def ASR(self, X, Y):
        _loss, _acc = self.model.evaluate(X, Y, verbose=0)
        return _acc


    def UF(self, x1, x2):
        probs1 = self.model(x1)
        probs2 = self.model(x2)
        z = np.max(probs1,axis=1)-np.max(probs2,axis=1)
        return np.average(np.abs(z))


    def CWA(self, x_c, y_c):
        _loss, _acc = self.model.evaluate(x_c, y_c, verbose=0)        
        return _acc


    def avg_values(self, x_val, layer_indexes):
        model = tf.keras.models.clone_model(self.initial_model)
        subModel = keras.Model(inputs=model.inputs, outputs=[model.layers[index].output for index in layer_indexes])
        internals = subModel(x_val)
        internals_avg = []
        for internal in internals:
            internal = internal.numpy()
            num_neurons = internal.shape[-1]
            internals_avg.append(np.average(internal.reshape(-1, num_neurons),axis=0))

        self.internals_avg = internals_avg
        return internals_avg


    def isolate_neurons(self, layer_indexes, neuron_indices, internals_avg):
        model = tf.keras.models.clone_model(self.initial_model)
        model.set_weights(self.initial_model.get_weights())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        for i in range(len(layer_indexes)): 
            layer_index = layer_indexes[i]
            neuron_index = neuron_indices[i]
            fixed_value = internals_avg[layer_index][neuron_index]
            weights, biases = model.layers[layer_index].get_weights()
            
            if isinstance(model.layers[layer_index], tf.keras.layers.Conv2D):
                weights[:, :, :, neuron_index] = 0.0  
                biases[neuron_index] = fixed_value
            else:
                weights[:, neuron_index] = 0.0  
                biases[neuron_index] = fixed_value 
                
            model.layers[layer_index].set_weights([weights, biases])
        self.model = model
        
        
    def neuron_estimition(self, layer_indexes, metric, X, Y):    
        NS = []

        if metric == 'ASR':
            self.metric = self.ASR
        elif metric == 'UF':
            self.metric = self.UF
        else:
            self.metric = self.CWA

        for layer_index in layer_indexes:
            l = []
            num_neurons = model.layers[layer_index].weights[0].numpy().shape[-1]
            for idx in range(num_neurons):
                self.model.isolate_neurons(layer_indexes=[layer_index], neuron_indices=[idx], internals_avg = self.internals_avg)
                l.append(self.metric(X,Y))
            NS.append(l)

        self.NS = NS
        return NS
    
    
    def topk(self, layer_indexes, NS, K):
        flattened_NS = np.concatenate(NS)
        topk_indexes = np.argsort(-flattened_NS)[:K]

        l_indexes = []
        n_indexes = []
        for i in layer_indexes:
            num_neurons = model.layers[i].weights[0].numpy().shape[-1]
            l_indexes.extend([i] * num_neurons)
            n_indexes.extend(range(0, num_neurons))

        NS_top_k_layer_indexes = [l_indexes[index] for index in topk_indexes]
        NS_top_k_neuron_indexes = [n_indexes[index] for index in topk_indexes]
        
        return NS_top_k_layer_indexes, NS_top_k_neuron_indexes
    
    
    def adapt(self, layer_indexes, NS, K, epsilon=0.1, A=20): 
        flattened_NS = np.concatenate(NS)
        topk_indexes = np.argsort(-flattened_NS)[:K]
        topm_indexes = np.argsort(-flattened_NS)[K:A]
        rands = np.random.random(K)
        num_utilize = np.sum(rands > epsilon)
        num_explore = np.sum(rands <= epsilon)
        utilize_indexes = random.sample(list(topk_indexes), num_utilize)
        explore_indexes = random.sample(list(topm_indexes), num_explore)
        greedy = utilize_indexes + explore_indexes

        l_indexes = []
        n_indexes = []
        for i in layer_indexes:
            num_neurons = model.layers[i].weights[0].numpy().shape[-1]
            l_indexes.extend([i] * num_neurons)
            n_indexes.extend(range(0, num_neurons))

        NS_adapt_layer_indexes = [l_indexes[index] for index in greedy]
        NS_adapt_neuron_indexes = [n_indexes[index] for index in greedy]
        
        return NS_adapt_layer_indexes, NS_adapt_neuron_indexes
        