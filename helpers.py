import itertools 
import numpy as np
import math
import matplotlib.pyplot as plt

grid_size = 101
grid = np.zeros((grid_size * grid_size,2))
nel = 0
for i,j in itertools.product(range(grid_size),range(grid_size)):
    x = (i - grid_size/2) * math.pi  / float(grid_size/2)
    y = (j - grid_size/2) * 8.0 / float(grid_size/2)
    grid[i*grid_size + j,:] = (x,y)

def plot_q(agent, episode = 0):
    action_scale = (agent.action_size - 1)/2
    preds = agent.model.predict(grid)
    preds -= np.min(preds)
    preds /= np.max(preds)
    preds_all = np.reshape(preds, [grid_size, grid_size, agent.action_size])
    extent = [np.min(grid,0)[0],np.max(grid,0)[0],np.min(grid,0)[1],np.max(grid,0)[1]]
    
    for i in range(agent.action_size):
        plt.subplot(2, agent.action_size, i+1)
    
        img = np.reshape(preds[:,i], [grid_size, grid_size]) - np.mean(preds_all,2)
        img = img - np.min(img)
        img = img / np.max(img)
        plt.imshow(img, extent=extent, aspect=0.5, interpolation='none', vmin = np.min(preds), vmax = np.max(preds))
        plt.title('Action %.1f' % (2.0*(i - action_scale)/action_scale))
        plt.ylim( ymax=np.max(grid,0)[1], ymin=np.min(grid,0)[1] )

    plt.subplot(2, agent.action_size, agent.action_size + 1)
    plt.imshow(np.argmax(preds_all,2), extent=extent, aspect=0.5, interpolation='none', vmin = np.min(preds), vmax = np.max(preds))
    plt.subplot(2, agent.action_size, agent.action_size + 2)
    plt.imshow(np.mean(preds_all,2), extent=extent, aspect=0.5, interpolation='none', vmin = np.min(preds), vmax = np.max(preds))
    plt.subplot(2, agent.action_size, agent.action_size + 3)
    plt.imshow(np.max(preds_all,2), extent=extent, aspect=0.5, interpolation='none', vmin = np.min(preds), vmax = np.max(preds))
    plt.savefig('frames/episode%05d.png' % episode)
