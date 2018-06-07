#
# Hyperparameter search by Bayesian Optimization
#

import os, shutil
import matplotlib.pyplot as plt
import matplotlib.colors
import pickle
import hypers
from bayes_opt import BayesianOptimization # https://github.com/fmfn/BayesianOptimization
import resnet
import vgg5
import vgg16
import evaluation

mcdir = 'models_computed'
bo = None
explored_settings = {}
vars_value_array = []
var_names = None
accuracies = []

def rm_dir_contents(dir_name):
    for the_file in os.listdir(dir_name):
        file_path = os.path.join(dir_name, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)    

def train_n_evaluate(resnet_activation_index, optimization_algorithm_index, 
                     batch_size_index, learning_rate_index, learning_rate_decay_index, epochs_index):
    
    global bo
    global explored_settings
    global vars_value_array
    global var_names
    global accuracies
    
    # set-up parameters (bo not so good for discrete values!)
    hypers.hyperparameters['resnet_activation'] = hypers.hyperparameter_values['resnet_activation'][int(resnet_activation_index)]
    hypers.hyperparameters['optimization_algorithm'] = hypers.hyperparameter_values['optimization_algorithm'][int(optimization_algorithm_index)]
    hypers.hyperparameters['batch_size'] = hypers.hyperparameter_values['batch_size'][int(batch_size_index)]
    hypers.hyperparameters['learning_rate'] = hypers.hyperparameter_values['learning_rate'][int(learning_rate_index)]
    hypers.hyperparameters['learning_rate_decay'] = hypers.hyperparameter_values['learning_rate_decay'][int(learning_rate_decay_index)]
    hypers.hyperparameters['epochs'] = hypers.hyperparameter_values['epochs'][int(epochs_index)] 
    
    if var_names is None:
        var_names = ['resnet_activation', 'optimization_algorithm', 'batch_size', 'learning_rate', 'learning_rate_decay', 'epochs']
    l_vva = []
    for v in var_names:
        l_vva.append(hypers.hyperparameters[v])
    vars_value_array.append(l_vva)
    
    settings_str = ','.join([k + ':' + str(hypers.hyperparameters[k]) for k in hypers.hyperparameters.keys()])
    if settings_str in explored_settings:
        print(' already explored this configuration')
        accuracy = explored_settings[settings_str]
    else:        
        # log output
        print(' training and evaluating with configuration:')
        print(str(hypers.hyperparameters))
        print('resnet_activation_index, optimization_algorithm_index, batch_size_index, learning_rate_index, learning_rate_decay_index, epochs_index')
        print([resnet_activation_index, optimization_algorithm_index, batch_size_index, learning_rate_index, learning_rate_decay_index, epochs_index])
        
        # train and save models with current parameters
        if not(os.path.exists("model")):
            os.makedirs('model')
        resnet.i_main(hypers.hyperparameters['epochs'], "model/resnet.h5")
        vgg5.i_main(hypers.hyperparameters['epochs'], "model/vgg5.h5")
        vgg16.i_main(hypers.hyperparameters['epochs'], "model/vgg16.h5")
        
        # evaluate current models
        accuracy = evaluation.i_main()
    
    # log parameters and accuracy
    log_line = [str(hypers.hyperparameters)]
    ns = 'resnet_activation_index,optimization_algorithm_index,batch_size_index,learning_rate_index,learning_rate_decay_index,epochs_index'.split(',')
    iv = [resnet_activation_index, optimization_algorithm_index, batch_size_index, learning_rate_index, learning_rate_decay_index, epochs_index]
    for n, i in zip(ns, iv):
        log_line.append(n + ':' + str(i))
    log_line.append('accuracy : ' + str(accuracy))
    with open("log.txt", "a") as myfile:
        myfile.write(','.join(log_line) + '\n')
    
    # remove previous models for next test
    if os.path.exists('model'):
        name_experiment = '-'.join(map(lambda x:str(x), iv))
        if not(os.path.exists(mcdir)):
            os.makedirs(mcdir)
            shutil.move('model', os.path.join(mcdir, name_experiment))
                
    # return accuracy
    print(' * Accuracy : ' + str(accuracy))
    explored_settings[settings_str] = accuracy
    accuracies.append(accuracy)
    return accuracy

def weithted_color(c1, c2, rate):
    c1_rgb = matplotlib.colors.to_rgb(c1)
    c2_rgb = matplotlib.colors.to_rgb(c2)
    result_color = tuple(int(c1x + rate * (c2x-c1x))
                         for c1x, c2x in zip(c1_rgb, c2_rgb))
    color = matplotlib.colors.to_hex(result_color)
    return color

def show_exploration(vars_value_array, vars_index_array, var_names, accuracies, title, xlabel, ylabel, file_name=None):
    colors = ['r', 'g', 'b', 'c']    

    fig = plt.figure()
    host = fig.add_subplot(111)
    par_dict = {}
    
    min_ac = min(accuracies)
    max_ac = max(accuracies)
    if vars_value_array is None:
        vars_value_array = []
        for s_ix in vars_index_array:
            vva_line = []
            for vi, s in zip(var_names, s_ix):
                s = int(float(s))
                print(vi + ' : ' + str(s))
                hv = hypers.hyperparameter_values[vi][s]
                vva_line.append(hv)
            vars_value_array.append(vva_line)
    alpha = 1.0 / float(len(vars_value_array))
    ndim = len(vars_value_array[0])
    for sample, accuracy in zip(vars_value_array, accuracies):
        if accuracy == max_ac:
            rate = 1.0
            print("best accuracy (" + str(accuracy) + "):" + str(sample))
        else:
            rate = (accuracy - min_ac) / (max_ac - min_ac)
        color = weithted_color(colors[1], colors[2], rate)
        sample_ix = []
        for vn, s in zip(var_names, sample):
            ix = hypers.hyperparameter_values[vn].index(s)
            sample_ix.append(ix)
        plt.plot(range(ndim), sample_ix, 'o-', color=color, alpha=alpha)    
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(range(ndim), var_names, rotation='vertical')
    plt.xlim(-1, ndim)
    plt.show()
    if not(file_name is None):
        plt.savefig(file_name, facecolor='#f8fafb')#, bbox_inches='tight')
        
def load_show():
    
    global bo
    global explored_settings
    global vars_value_array
    global var_names
    global accuracies
    

    with open("all.pkl", 'rb') as handle:
        obj_all = pickle.load(handle)
    
    vars_value_array = obj_all['vars_value_array'] 
    var_names = obj_all['var_names'] 
    accuracies = obj_all['accuracies'] 
    
    show_exploration(vars_value_array, None, var_names, accuracies, 'combinations explored\nand accuracy', 'variables', 'values', 'bo-search.png')    
    
    return

def main():
    
    global bo
    global explored_settings
    global vars_value_array
    global var_names
    global accuracies
    
    if os.path.isfile("all.pkl"):
        os.unlink("all.pkl")
    if os.path.isfile("log.txt"):
        os.unlink("log.txt")
    if os.path.isfile("bo-log.txt"):
        os.unlink("bo-log.txt")
    if os.path.isfile("es-log.txt"):
        os.unlink("es-log.txt")
    if os.path.exists("model"):
        rm_dir_contents('model')
    if os.path.exists(mcdir):
        rm_dir_contents(mcdir)
    
    hyper_vars = hypers.hyperparameters.keys()
    hyper_var_indexes = [len(hypers.hyperparameter_values[x])-1 for x in hyper_vars]
    vars_indexes_dict = {}
    for v, il in zip(hyper_vars, hyper_var_indexes):
        vars_indexes_dict[v] = (0, il) # 0 to num of values of hyper parameter
    
    bo = BayesianOptimization(lambda resnet_activation, optimization_algorithm, batch_size, learning_rate, learning_rate_decay, epochs: train_n_evaluate(resnet_activation, optimization_algorithm, batch_size, learning_rate, learning_rate_decay, epochs), vars_indexes_dict)
    
    bo.maximize(init_points=3, n_iter=5, kappa=2)
    
    obj_all = {}
    obj_all['bo-max'] = bo.res['max']
    obj_all['bo-all'] = bo.res['all']
    obj_all['explored_settings'] = explored_settings
    obj_all['vars_value_array'] = vars_value_array
    obj_all['var_names'] = var_names
    obj_all['accuracies'] = accuracies
    with open("all.pkl", 'wb') as handle:
        pickle.dump(obj_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('explored settings')
    print(str(explored_settings))
    with open("es-log.txt", "a") as myfile:
        myfile.write(str(explored_settings) + '\n') 
    
    print("max")
    print(bo.res['max'])
    print("all")
    print(bo.res['all'])
    
    show_exploration(vars_value_array, None, var_names, accuracies, 'combinations explored\nand accuracy', 'variables', 'values', 'bo-search.png')
    
    return

if __name__ == '__main__':
    #load_show()
    main()
    
    
    