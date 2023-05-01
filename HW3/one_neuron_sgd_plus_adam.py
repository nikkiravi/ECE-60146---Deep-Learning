# Import Libraries
import random
import numpy as np
import matplotlib.pyplot as plt
import operator
from ComputationalGraphPrimer import *

# Constants
SEED = 512           
random.seed(SEED)
np.random.seed(SEED)


class ComputationalGraphPrimerSGDPlus(ComputationalGraphPrimer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Inheriting from the parent class

    # Modifying and Overriding the run_training_loop_one_neuron_model to implement SGD+
    # mu is between [0,1]
    def run_training_loop_one_neuron_model(self, training_data, mu=0.5):
        #########################################Copied from the original function###################################
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params} # initializing learnable parameters with random numbers from a uniform distribution over the interval (0,1)

        self.bias = random.uniform(0,1)                   ## Adding the bias improves class discrimination.
                                                          ##   We initialize it to a random number.

        class DataLoader:
            """
            To understand the logic of the dataloader, it would help if you first understand how 
            the training dataset is created.  Search for the following function in this file:

                             gen_training_data(self)
           
            As you will see in the implementation code for this method, the training dataset
            consists of a Python dict with two keys, 0 and 1, the former points to a list of 
            all Class 0 samples and the latter to a list of all Class 1 samples.  In each list,
            the data samples are drawn from a multi-dimensional Gaussian distribution.  The two
            classes have different means and variances.  The dimensionality of each data sample
            is set by the number of nodes in the input layer of the neural network.

            The data loader's job is to construct a batch of samples drawn randomly from the two
            lists mentioned above.  And it mush also associate the class label with each sample
            separately.
            """
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]   ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]   ## Associate label 1 with each sample

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                            ## When a batch is created by getbatch(), we want the
                                                                           ##   samples to be chosen randomly from the two lists
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                            ## First list for samples, the second for labels
                maxval = 0.0                                               ## For approximate batch data normalization
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]          ## Normalize batch data
                batch = [batch_data, batch_labels]
                return batch                

        #############################################################################################################
        # Modified part of the function
        self.mu = mu
        self.bias_factor = 0 # Update the bias, the factor depends on the current mu
        self.step_sizes = [0 for i in range(len(self.learnable_params) + 1)]

        #########################################Copied from the original function###################################
        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                    ##  Average the loss over iterations for printing out 
                                                                           ##    every N iterations during the training loop.
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples)              ##  FORWARD PROP of data
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])  ##  Find loss
            loss_avg = loss / float(len(class_labels))                                              ##  Average the loss over batch
            avg_loss_over_iterations += loss_avg                          
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                 ## Display average loss
                avg_loss_over_iterations = 0.0                                                     ## Re-initialize avg loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(operator.truediv, data_tuple_avg, 
                                     [float(len(class_labels))] * len(class_labels) ))
            self.backprop_and_update_params_one_neuron_model(y_error_avg, data_tuple_avg, deriv_sigmoid_avg)     ## BACKPROP loss
        # plt.figure()     
        # plt.plot(loss_running_record) 
        # plt.show()   
        return loss_running_record

        #############################################################################################################

    # Modify backpropagation function for one_neuron_model - backpropagating the loss and updating the values of the learnable parameters.
    def backprop_and_update_params_one_neuron_model(self, y_error, vals_for_input_vars, deriv_sigmoid):
        """
        As should be evident from the syntax used in the following call to backprop function,

           self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                     ^^^             ^^^                ^^^
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.

        See Slide 59 of my Week 3 slides for the math of back propagation for the One-Neuron network.
        """
        input_vars = self.independent_vars
        vals_for_input_vars_dict = dict(zip(input_vars, list(vals_for_input_vars)))
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
            ## Calculate the next step in the parameter hyperplane
            #################################################Modified##########################################
            self.step_sizes[i + 1] = (self.mu * self.step_sizes[i]) + (self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid)  
            self.step_sizes[i] = self.step_sizes[i + 1] # Save the new current step size in the previous iteration position
            
            ## Update the learnable parameters
            # self.vals_for_learnable_params[param] += -self.learning_rate * self.step_sizes[i + 1]
            self.vals_for_learnable_params[param] += self.step_sizes[i + 1]
        
        self.bias_factor = (self.mu * self.bias_factor) + (self.learning_rate * y_error * deriv_sigmoid) ## Update the bias
        self.bias += self.bias_factor
    ######################################################################################################

class ComputationalGraphPrimerAdam(ComputationalGraphPrimer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Modifying and Overriding the run_training_loop_one_neuron_model to implement SGD+
    # Beta1 and Beta2 are close to 1
    def run_training_loop_one_neuron_model(self, training_data, beta1=0.9, beta2=0.999, e=1e-6):
        #########################################Copied from the original function###################################
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params} # initializing learnable parameters with random numbers from a uniform distribution over the interval (0,1)

        self.bias = random.uniform(0,1)                   ## Adding the bias improves class discrimination.
                                                          ##   We initialize it to a random number.

        class DataLoader:
            """
            To understand the logic of the dataloader, it would help if you first understand how 
            the training dataset is created.  Search for the following function in this file:

                             gen_training_data(self)
           
            As you will see in the implementation code for this method, the training dataset
            consists of a Python dict with two keys, 0 and 1, the former points to a list of 
            all Class 0 samples and the latter to a list of all Class 1 samples.  In each list,
            the data samples are drawn from a multi-dimensional Gaussian distribution.  The two
            classes have different means and variances.  The dimensionality of each data sample
            is set by the number of nodes in the input layer of the neural network.

            The data loader's job is to construct a batch of samples drawn randomly from the two
            lists mentioned above.  And it mush also associate the class label with each sample
            separately.
            """
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]   ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]   ## Associate label 1 with each sample

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                            ## When a batch is created by getbatch(), we want the
                                                                           ##   samples to be chosen randomly from the two lists
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                            ## First list for samples, the second for labels
                maxval = 0.0                                               ## For approximate batch data normalization
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]          ## Normalize batch data
                batch = [batch_data, batch_labels]
                return batch                

        #############################################################################################################
        # Modified part of the function
        self.beta1, self.beta2 = beta1, beta2
        self.e = e
        self.m_db, self.v_db = 0, 0
        self.m_dw = [0 for i in range(len(self.learnable_params) + 1)] # m
        self.v_dw = [0 for i in range(len(self.learnable_params) + 1)] # v


        #########################################Copied from the original function###################################
        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                    ##  Average the loss over iterations for printing out 
                                                                           ##    every N iterations during the training loop.
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples)              ##  FORWARD PROP of data
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])  ##  Find loss
            loss_avg = loss / float(len(class_labels))                                              ##  Average the loss over batch
            avg_loss_over_iterations += loss_avg                          
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                 ## Display average loss
                avg_loss_over_iterations = 0.0                                                     ## Re-initialize avg loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(operator.truediv, data_tuple_avg, 
                                     [float(len(class_labels))] * len(class_labels) ))
            self.backprop_and_update_params_one_neuron_model(y_error_avg, data_tuple_avg, deriv_sigmoid_avg, i + 1)     ## BACKPROP loss
        # plt.figure()     
        # plt.plot(loss_running_record) 
        # plt.show()   

        return loss_running_record

        #############################################################################################################

    # Modify backpropagation function for one_neuron_model - backpropagating the loss and updating the values of the learnable parameters.
    def backprop_and_update_params_one_neuron_model(self, y_error, vals_for_input_vars, deriv_sigmoid, k):
        """
        As should be evident from the syntax used in the following call to backprop function,

           self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                     ^^^             ^^^                ^^^
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.
    
        See Slide 59 of my Week 3 slides for the math of back propagation for the One-Neuron network.
        """
        input_vars = self.independent_vars
        vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
            ## Calculate the next step in the parameter hyperplane
            #################################Modified##################################
            self.m_dw[i + 1] = (self.beta1 * self.m_dw[i]) + ((1 - self.beta1) * (self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid))
            self.m_dw[i] = self.m_dw[i + 1] # Save the new current moment in the previous iteration position
            
            self.v_dw[i + 1] = (self.beta2 * self.v_dw[i]) + ((1 - self.beta2) * (self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid)**2)
            self.v_dw[i] = self.v_dw[i + 1] # Save the new current moment in the previous iteration position
            
            ## Update the learnable parameters 
            mk_hat = self.m_dw[i + 1] / (1 - self.beta1 ** k) 
            vk_hat = self.v_dw[i + 1] / (1 - self.beta2 ** k)

            self.vals_for_learnable_params[param] += mk_hat / np.sqrt(vk_hat + self.e)
        
        # Inspired by: https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
        # Inspired by: https://www.youtube.com/watch?v=JXQT_vxqwIs&ab_channel=DeepLearningAI
        self.m_db = (self.beta1 * self.m_db) + (1 - self.beta1) * (self.learning_rate * y_error * deriv_sigmoid)
        self.v_db = (self.beta2 * self.v_db) + (1 - self.beta2) * (self.learning_rate * y_error * deriv_sigmoid) ** 2

        m_db_hat = self.m_db / (1 - self.beta1 ** k) 
        v_db_hat = self.v_db / (1 - self.beta1 ** k) 

        self.bias += m_db_hat / np.sqrt(v_db_hat + self.e) ## Update the bias
    ######################################################################################################

def sgd_plus(lr=1e-3, mu=0.9):
    cgp = ComputationalGraphPrimerSGDPlus(
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
               dataset_size = 5000,
               learning_rate = lr,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )

    cgp.parse_expressions()
    # cgp.display_one_neuron_network()      

    training_data = cgp.gen_training_data()
    loss_per_iteration = cgp.run_training_loop_one_neuron_model( training_data, mu=mu)

    return loss_per_iteration

def adam(lr=1e-3):
    cgp = ComputationalGraphPrimerAdam(
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
               dataset_size = 5000,
               learning_rate = lr,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )

    cgp.parse_expressions()
    # cgp.display_one_neuron_network()      

    training_data = cgp.gen_training_data()
    loss_per_iteration = cgp.run_training_loop_one_neuron_model( training_data )

    return loss_per_iteration

def sgd(lr=1e-3):
    cgp = ComputationalGraphPrimerSGDPlus(
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
               dataset_size = 5000,
               learning_rate = lr,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
    )

    cgp.parse_expressions()
    # cgp.display_one_neuron_network()      

    training_data = cgp.gen_training_data()
    loss_per_iteration = cgp.run_training_loop_one_neuron_model( training_data, mu=0)

    return loss_per_iteration

def plot_losses(sgd, sgd_plus, adam, lr):
    number_of_iterations = len(adam)
    plt.plot(range(number_of_iterations), sgd, label="SGD Loss")
    plt.plot(range(number_of_iterations), sgd_plus, label="SGD+ Loss")
    plt.plot(range(number_of_iterations), adam, label="Adam Loss")

    plt.title(f"Loss per Iteration for Different Optimizers for One Neuron Model for Learning Rate: {lr}")
    plt.xlabel("Iteration Number")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")

    # plt.show(); quit()
    plt.savefig(r"/Users/nikitaravi/Documents/Academics/ECE 60146/HW3/one_neuron_" + str(lr) + "_learning_rate.png", dpi=200)

if __name__ == "__main__":
    lr = 1e-3
    sgd_loss = sgd(lr)
    sgd_plus_loss = sgd_plus(lr)
    adam_loss = adam(lr)

    plot_losses(sgd_loss, sgd_plus_loss, adam_loss, lr)