import math
import random
import numpy as np
from Chatbot.embedding import embedding
def positional_encoder(orignal_data:str):
    tokens = orignal_data.split(" ")
    encoded = []
    for i,j in enumerate(tokens):
        this_token_encoded = []
        pos = i
        for n in range(100):
            value = pos/(10000**((2*n)/100))
            if n%2 ==0:
                en = math.sin(value)
            else:
                en = math.cos(value)
            this_token_encoded.append(en)
        encoded.append(this_token_encoded)
    return encoded
def mean(v):
    sum = 0
    for i in v:
        sum = sum + i
    return sum/len(v)
def std(p):
    m = mean(p)
    dif = []
    dif_sqr = []
    for i in p:
        dif.append(i-m)
    for j in dif:
        dif_sqr.append(j**2)
    r = sum(dif_sqr)/len(p)
    final = math.sqrt(r)
    return final if r!=0 else 1e-8
class layernorm:
    def __init__(self,output):
        y = [1 for _ in range(len(output[0]))]
        b = [0 for _ in range(len(output[0]))]
        norm = []
        for i in output:
            avg = mean(i)
            stddev = std(i)
            this_norm = []
            for j in i:
                prd = (j-avg)/stddev+1e-15
                this_norm.append(prd)
            this = []#np.array(y)*np.array(this_norm)+b
            for n,j,k in zip(this_norm,y,b):
                this.append(j*n+k)
            norm.append(this)
        self.norm = norm
        self.y = y
        self.b=b
class jeevan:
    def train(self,data:str):
        embedder= embedding(data)
        embedder.train()
        embeded = embedder.get_embedded()
        list_embeded = []
        for i in embeded:
            list_embeded.append(embeded[i])
        positionl_encoed = positional_encoder(data)
        z = []
        for i,j in zip(list_embeded,positionl_encoed):
            this = []
            for m,n in zip(i,j):
                this.append(m+n)
            z.append(this)
        h = 10
        k = 100/10
        outputs = []
        q_w = []
        k_w = []
        v_w =[]
        L = len(z)
        mask = np.triu(np.ones((L, L)) * float('-inf'), k=1)
        for som in range(h):
            this_q_w = []
            this_k_w = []
            this_v_w = []
            for i in range(100):
                this_q_w.append([random.random() for _ in range(int(k))])
                this_k_w.append([random.random() for _ in range(100)])
                this_v_w.append([random.random() for _ in range(100)])
            q_w.append(this_q_w)
            k_w.append(this_k_w)
            v_w.append(this_v_w)
            q_arr = np.array(this_q_w)@np.array(z)
            k_arr = np.array(this_k_w)@np.array(z)
            v_arr = np.array(this_v_w)@np.array(z)
            k_t = k_arr.transpose()
            a = (q_arr@k_t)+mask
            attention = (a/math.sqrt(k)).tolist()
            act  = []
            for variable in attention:
                exponated = [math.exp(value) for value in variable]
                sum_of_exponated = sum(exponated)
                activated = [raw/sum_of_exponated for raw in exponated]
                act.append(activated)
            final = np.array(act)@v_arr
            outputs.append(final)
        mlti_head = np.concatenate(outputs,axis=1)
        w_O = []
        for i in range(100):
            w_O.append([random.random() for _ in range(100)])
        project = ((mlti_head@np.array(w_O))+z).tolist()
        attention_norm  = layernorm(project)
        project = layernorm(project).norm  
        ouptut_of_layer_1 = []
        weights_of_layer1 = []
        bias_of_layer1 = []
        for i in range(0,1):
            inputv = project
            weights_for_layer = []
            bias_for_layer = []
            output_of_layer = []
            for j in range(400):
                this_neuron_output = []
                weights_for_neuron = [random.random() for _ in range(100)]
                bias_for_neuron = random.random()
                for each in inputv:
                    p = 0
                    for w,inp in zip(weights_for_neuron,each):
                        p+=w*inp
                    p+=bias_for_neuron
                    this_neuron_output.append(p)
                output_of_layer.append(this_neuron_output)
                bias_for_layer.append(bias_for_neuron)
                weights_for_layer.append(weights_for_neuron)
            ouptut_of_layer_1.append(output_of_layer)
            weights_of_layer1.append(weights_for_layer)
            bias_of_layer1.append(bias_for_layer)
        for i,j in enumerate(ouptut_of_layer_1):
            for k,l in enumerate(j):
                for e,f in enumerate(l):
                    ouptut_of_layer_1[i][k][e] = max(0,f)
        output_of_layer2 = []
        weights_of_layers2 = []
        bias_of_layer2 = []
        for i in range(0,1):
            inputv = ouptut_of_layer_1
            weights_for_layer = []
            bias_for_layer = []
            output_of_layer = []
            for j in range(100):
                this_neuron_output = []
                weights_for_neuron = [random.random() for _ in range(100)]
                bias_for_neuron = random.random()
                for each in inputv:
                    p = 0
                    for w,inp in zip(weights_for_neuron,each):
                        p+=w*inp
                    p+=bias_for_neuron  
                    this_neuron_output.append(p)
                weights_for_layer.append(weights_for_neuron)
                output_of_layer.append(this_neuron_output)
                bias_for_layer.append(bias_for_neuron)
            output_of_layer2.append(output_of_layer)
            weights_of_layers2.append(weights_for_layer)
            bias_of_layer2.append(bias_for_layer)
        output_of_layer2 = (np.array(output_of_layer2)+np.array(z)).tolist()
        output_of_layer2 = layernorm(output_of_layer2).norm
        