import random
import pickle
import math
class embedding:
    def __init__(self,data:str,seperator_value=" "):
        tokens = data.split(seperator_value)
        unique_tokens = list(dict.fromkeys(tokens))
        ids = []
        for i in range(len(unique_tokens)):
            ids.append(i)
        token_id_map = {}
        for i,j in zip(unique_tokens,ids):
            token_id_map.update({i:j})
            all_word_vector = []
        main_m = {}
        main_v = {}
        for i in ids:
            main_m.update({i:[0 for _ in range(100)]})
            main_v.update({i:[0 for _ in range(100)]})
        for i in range(len(ids)):
            this_word_vector = [random.random() for _ in range(100)]
            all_word_vector.append(this_word_vector)
        vector_token_map = {}
        for i,j in zip(unique_tokens,all_word_vector):
            vector_token_map.update({i:j})
        self.vector_token = vector_token_map
        self.unique = unique_tokens
        self.ids = ids
        self.token_id = token_id_map
        self.tokens = tokens
        self.m = main_m
        self.v = main_v
    def train(self,epoch=100,beta1=0.9,beta2=0.9999,epsilon=1e-8,learning_rate=0.01,save=True):
        update_count = 0
        for som in range(epoch):
            for t in range(len(self.tokens)-1):
                print(f"starting pair {self.tokens[t]} and {self.tokens[t+1]} in epoch {som}")
                update_count+=1
                inputv = self.tokens[t]
                output = self.tokens[t+1]
                i = self.vector_token[inputv]
                j = self.vector_token[output]
                dot = []
                output_V = 0
                for lkj in self.vector_token:
                    this = self.vector_token[lkj]
                    if lkj==inputv:
                        continue
                    elif lkj==output:
                        output_weighted = [asdf*n for asdf,n in zip(this,i)]
                        output_V+=sum(output_weighted)
                        dot.append(sum(output_weighted))
                    else:
                        weighted = [asdf*n for asdf,n in zip(this,i)]
                        dot.append(sum(weighted))
                exponated = []
                max_dot = max(dot)
                output_exp = math.exp(output_V-max_dot)
                for n in dot:
                    exponated.append(math.exp(n-max_dot))
                sum_of_exponated = sum(exponated)
                activated = []
                output_act = output_exp/sum_of_exponated
                for n in exponated:
                    activated.append(n/sum_of_exponated)
                actual = output_act  
                one_hot = []          
                for s in activated:
                    if s==actual:
                        one_hot.append(1)
                    else:
                        one_hot.append(0)
                g = []
                for s,so in zip(activated,one_hot):
                    g.append(s-so)
                mi = []
                for s,k in zip(self.vector_token,g):
                    token = self.vector_token[s]
                    multiplied = [v*k for v in token]
                    mi.append(multiplied)
                gradient = []
                for s in zip(*mi):
                    gradient.append(sum(s))
                for d in range(100):
                    current_id = self.token_id[inputv]
                    self.m[current_id][d] = beta1*self.m[current_id][d]+(1-beta1)*gradient[d]
                    self.v[current_id][d] = beta2*self.v[current_id][d]+(1-beta2)*(gradient[d]**2)
                update = []
                correct_m = []
                correct_v = []
                for d in range(100):
                    current_id = self.token_id[inputv]
                    m_val = self.m[current_id][d]
                    v_val = self.v[current_id][d]
                    corrected_M = m_val/(1-beta1**update_count)
                    corrected_v = v_val/(1-beta2**update_count)
                    correct_m.append(corrected_M)
                    correct_v.append(corrected_v)
                for d in range(100):
                    update.append(correct_m[d]/(math.sqrt(correct_v[d])+epsilon))
                for d in range(100):
                    self.vector_token[inputv][d] -= learning_rate*update[d]
                output_id = self.token_id[output]
                for d in range(100):
                    self.m[output_id][d] = beta1*self.m[output_id][d]+(1-beta1)*gradient[d]
                    self.v[output_id][d] = beta2*self.v[output_id][d]+(1-beta2)*(gradient[d]**2)
                corrected_m_out = []
                corrected_v_out = []
                for d in range(100):
                    m_val = self.m[output_id][d]
                    v_val = self.v[output_id][d]
                    co_m = m_val/(1-beta1**update_count)
                    co_v = v_val/(1-beta2**update_count)
                    corrected_m_out.append(co_m)
                    corrected_v_out.append(co_v)
                update_out = []
                for d in range(100):
                    update_out.append(corrected_m_out[d]/(math.sqrt(corrected_v_out[d])+epsilon))
                for d in range(100):
                    self.vector_token[output][d] -= learning_rate*update_out[d]
        with open("cache.pkl","ab") as file:
            pickle.dump(self.vector_token,file)
    def get_embedded(self):
        with open("cache.pkl","rb") as f:
            return pickle.load(f)

