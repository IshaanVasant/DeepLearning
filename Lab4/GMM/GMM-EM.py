import numpy as np
import matplotlib.pyplot as plt

def plot_data(data, gaussian_labels):
    unique_gaussian= np.unique(gaussian_labels)
    
    plt.figure()
    
    for i in range(len(unique_gaussian)):
        gauss = [data[j] for j in range(len(data)) if gaussian_labels[j] == unique_gaussian[i]]
        gauss = np.array(gauss)
        plt.hist(gauss, bins=100)

def gauss_prob(x, w, m, variance):
    return (w/np.sqrt(2*np.pi*variance)) * np.exp(-((x-m)**2)/(2*variance))


def generate_data():     
    
    m = [-2, 0.5, 2, 4.2, 7]
    variance = [0.18, 0.13, 0.4, 0.5, 0.8]
    n_gauss = len(m)  
    n_data_points = 1000   
    data = np.array([])
    for i in range(n_gauss):
        data = np.append(data, np.random.normal(m[i], np.sqrt(variance[i]), n_data_points)) 
    
    return data, n_gauss
    

def EM(data, n_gauss, gaussian_labels, w_pred, m_pred, variance_pred):
    
    prob = np.zeros((len(data), n_gauss))
    n_iter = 100
    for i in range(n_iter):
        # Expectation 
        for j in range(len(data)):
            prob[j] = gauss_prob(data[j], w_pred, m_pred, variance_pred)
            prob[j] /=np.sum(prob[j])
            gaussian_labels[j] = np.argmax(prob[j])   
        
        # Maximization    
        wts = np.multiply(data.T, prob.T).T
        m_pred = np.sum(wts, axis = 0)/np.sum(prob, axis = 0)
        w_pred = np.sum(prob, axis=0)/len(data)        
     
        for j in range(n_gauss):
            total_variance = 0
            for k in range(len(data)):
                total_variance += prob[k, j] * ((data[k] - m_pred[j])**2)            
            variance_pred[j] = total_variance/ np.sum(prob[:,j])
            
    return m_pred, variance_pred, w_pred, gaussian_labels
    
if __name__ == "__main__":
    
    data, n_gauss = generate_data()
    m = data[np.random.randint(0, len(data), n_gauss)]
    variance = np.random.rand(1, n_gauss)[0]
    wt = np.random.rand(1, n_gauss)[0]
    wt/=(np.sum(wt))
    gaussian_labels = np.random.randint(0, 3, (len(data),1))
    plot_data(data, gaussian_labels)
    m_pred, variance_pred, w_pred, gaussian_labels = EM(data, n_gauss, gaussian_labels,  wt, m, variance)
    plot_data(data, gaussian_labels)
    num = float(input('Enter random data point between (-3, 8): '))
    prob_pred = gauss_prob(num, w_pred, m_pred, variance_pred)
    prob_pred /= np.sum(prob_pred)
    print('Predicted Probabilities: ', prob_pred)