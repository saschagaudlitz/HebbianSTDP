"""

Simulates and visualises Hebbian STDP. Both a single weight vector and multiple weight vectors
are considered. Allows for correlations.

"""

using PyPlot           # For visualization
using LinearAlgebra    # For matrix operations and norms
using Distributions    # For random sampling (Categorical, Uniform)
using Random           # For reproducible results

# Standard plot parameters
standard_colors = ["black", "blue", "red"]  # Colors for different neuron curves
standard_fontsize = 13                      # Standard font size for plots

#====================================================================================================#
# CONFIGURATION PARAMETERS
#====================================================================================================#

# Probability simplex configuration 
polygon_points = [[0,0], [1,0], [1/2,sqrt(3)/2]]  # Vertices of probability simplex (equilateral triangle)
N = 1000                                          # Grid resolution for plotting
x = LinRange(0,1,N)                               # x-coordinates for grid underlying the simplex
y = LinRange(0,sqrt(3)/2,N)                       # y-coordinates for grid underlying the simplex

# Single weight vector learning parameters
lr = 10^(-2)                                      # Learning rate for single vector
number_time_steps = 20*round(Int, 1/lr)           # Number of simulation steps

# Trajectory simulation parameters
n_trajectories = 3                             # Number of trajectories to simulate
if n_trajectories == 3
    starting_points = [[0.25, 0.4, 0.35],           # Starting points for n_trajectories = 3
                        [0.40, 0.27, 0.33], 
                        [0.22, 0.38, 0.40]]
else
    static_start_point = [0.3, 0.3, 0.4]            # Fixed starting point for many runs (n_trajectories >> 3)
    starting_points = [static_start_point for _ in 1:n_trajectories]
end
correlation_matrix = [1.0 0.0 0.0;               # Correlation matrix (Gamma) for input neurons
                      0.0 1.0 0.0; 
                      0.0 0.0 1.0]


# Multiple weight vectors learning parameters
M = 3                                             # Number of input neurons
K = 3                                             # Number of output neurons (weight vectors)
intensities = [10.,7.5,5.]                        # Input neuron intensities (highest to lowest)
n_runs = 1                                        # Number of simulation runs
lr_base = 10^(-3)                                 # Base learning rate
rel_lr_scaling = [1., .75, .5]                    # Relative learning rate for each output neuron
number_time_steps_multiple_weights = 4*10^4       # Simulation steps for multiple weights
weights_init = ones(K,M) / M                      # Initial weights (uniform distribution)

Random.seed!(5)                                   # Set random seed for reproducibility
#====================================================================================================#


"""
Computes the loss landscape on the probability simplex.
Arguments:
- x, y: Coordinate grids
- grad: Whether to plot gradient flow (default: true)
- plotting: Whether to visualize results (default: true)
Returns: Matrix of loss values
"""
function LossLandscape(x, y; grad = true, plotting = true)
    # Compute loss values across the grid
    z = zeros(length(x), length(y))
    
    for i in 1:length(x)
        for j in 1:length(y)
            bary = cart_to_bary([x[i], y[j]])
            z[i, j] = minimum(bary) < 0 ? NaN : L(bary)  # NaN for points outside simplex
        end
    end
    
    # Create basic contour plot of loss landscape
    if plotting
        figure()
        create_standard_contour(x, y, z)
        create_standard_colorbar(z)
        title("Loss function on the probability simplex")
        tight_layout()
    end

    # Add gradient flow visualization
    if grad
        grad_x = zeros(length(x), length(y))
        grad_y = zeros(length(x), length(y))
        
        for i in 2:length(x)-1
            for j in 2:length(y)-1
                bary = cart_to_bary([x[i], y[j]])
                if minimum(bary) < 0
                    grad_x[i, j] = grad_y[i,j] = NaN  # NaN for points outside simplex
                else
                    grad_cart = gradient_bary_to_cart(gradientL(bary))
                    grad_x[i,j] = grad_cart[1]
                    grad_y[i,j] = grad_cart[2]
                    #grad_x[i,j], grad_y[i,j] = bary_to_cart(gradientL(bary))
                end
            end
        end
        
        figure()
        create_standard_contour(x, y, z)
        create_standard_colorbar(z)
        streamplot(x',y',grad_x',grad_y', density=1, color="black", 
                  broken_streamlines=false, linewidth=0.5, arrowsize=0.5)
        tight_layout()
    end
    
    return z
end

"""
Simulates individual Hebbian STDP trajectories on the probability simplex.
Arguments:
- p_0: Initial point in barycentric coordinates
- lr: Learning rate 
- num_steps: Number of simulation steps
Returns: Tuple of (cartesian trajectory, barycentric trajectory)
"""
function indiv_trajectory(p_0, lr, num_steps)
    # Pre-allocate arrays
    traject = zeros(num_steps, 2)        # Trajectory in cartesian coordinates
    traject_bary = zeros(num_steps, 3)   # Trajectory in barycentric coordinates
    
    # Initialise starting point
    traject[1, :] = bary_to_cart(p_0)
    traject_bary[1, :] = p_0
    
    for i in 1:num_steps-1
        bary_old = traject_bary[i, :]
        
        # Generate STDP updates:
        # B: One-hot vector with random index based on current probabilities
        # Z: Uniform random noise

 
        C = rand.(Bernoulli.(correlation_matrix))
        B = rand.(Bernoulli.(min.(C * bary_old,1)))
        #B = zeros(3)
        #B[rand(Categorical(bary_old))] = 1
        Z = rand(Uniform(-1,1), 3)
        
        #Update step
        #normaliser = 1 + lr * sum(bary_old .* (B + Z))
        #bary_new = @. bary_old * (1 + lr * (B + Z)) / normaliser
        bary_new = @. bary_old * (1 + lr * (B + Z))
        bary_new /= sum(bary_new)  # Project back onto simplex

        
        traject_bary[i+1, :] = bary_new
        traject[i+1, :] = bary_to_cart(bary_new)
    end
    
    return traject, traject_bary
end

"""
Plots multiple STDP trajectories on the loss landscape.
Arguments:
- x, y: Coordinate grids
- starting_points: Array of initial points in barycentric coordinates
- lr: Learning rate
- number_time_steps: Number of simulation steps
"""
function plot_trajectories(x,y,starting_points, lr, number_time_steps)
    # Get loss landscape without plotting
    z = LossLandscape(x, y; grad = false, plotting = false)
    
    figure()
    create_standard_contour(x, y, z)
    create_standard_colorbar(z)

    # Compute all trajectories
    all_trajectories = [indiv_trajectory(starting_points[i], lr, number_time_steps) 
                        for i in 1:length(starting_points)]
    
    # Plot trajectories with different styles based on number of trajectories
    if length(starting_points) <= 3
        # Few trajectories: use different colors and labels
        my_colors = ["red", "black", "blue"]
        for i in 1:length(starting_points)
            sp = starting_points[i]
            label = L"\mathbf{p}(0)=" * "(" * string(round(sp[1], digits=2)) * "," * 
                    string(round(sp[2], digits=2)) * "," * string(round(sp[3], digits=2)) * ")"
            
            plot(all_trajectories[i][1][:,1], all_trajectories[i][1][:,2], 
                 color=my_colors[i], label=label)
        end
        legend(fontsize=standard_fontsize)
    else
        # Many trajectories: plot all in black with thin lines
        for i in 1:length(starting_points)
            plot(all_trajectories[i][1][:,1], all_trajectories[i][1][:,2], 
                 color="black", linewidth=0.2, alpha=1.0)
        end
    end
    
    tight_layout()
end

"""
Simulates learning dynamics for multiple weight vectors with orthogonalisation.
Arguments:
- weights_init: Initial weight matrix (K×M)
- lr: Base learning rate
- number_time_steps: Number of simulation steps
- intensities: Input neuron intensity values
- rel_lr_scaling: Relative learning rate scaling for each output neuron
- plotting: Whether to visualize results (default: true)
Returns: Probability tensor (time_steps × K × M)
"""
function learning_dynamics(weights_init, lr, number_time_steps, intensities, rel_lr_scaling; plotting=true)
    # Pre-allocate weights tensor (time × output neurons × input neurons)
    weights = zeros(number_time_steps, K, M)
    weights[1, :, :] = weights_init
    
    for i in 1:number_time_steps-1
        for k in 1:K
            curr_weights = @view weights[i, k, :]
            
            # Calculate weighted probabilities based on intensities
            probabilities = curr_weights .* intensities / sum(curr_weights .* intensities)
            
            # Generate STDP updates, the data B and Z are given by the network and provided to the statistician
            B = zeros(M)
            B[rand(Categorical(probabilities))] = 1  # Sample based on probability
            Z = rand(Uniform(-1,1), M)               # Random noise
            
            # Update weights with learning rate scaling
            delta = @. curr_weights * lr * rel_lr_scaling[k] * (B + Z)
            
            # Orthogonalise against previous weight vectors (Gram-Schmidt process)
            for j in 1:(k-1)
                prev_weights = @view weights[i, j, :]
                delta -= sum(delta .* prev_weights) / sum(prev_weights.^2) * prev_weights
            end
            
            # Apply update and ensure non-negativity; the latter only needed because of numeric instability where values like -10^(-10) can occur
            weights[i+1, k, :] = max.(0, curr_weights + delta)
        end
    end
    
    # Convert weights to probabilities
    my_probs = weights .* reshape(intensities, 1, 1, :)
    my_probs = my_probs ./ sum(my_probs, dims=3)
    
    # Visualise probability evolution
    if plotting
        figure(figsize=(15, 5))
        lw = 1.5
        for k in 1:K
            subplot(1, 3, k)
            for m in 1:M
                plot(1:number_time_steps, my_probs[:, k, m], 
                     label=L"p_{%$m%$k}", color=standard_colors[m], linewidth=lw)
            end
            title(L"Output neuron %$k")
            xlabel("Iteration index")
            ylabel("Probability")
            legend()
            gca().ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=true)
        end
        tight_layout()
    end
    
    return my_probs
end

"""
Visualises the trajectories of multiple weight vectors on the probability simplex.
Arguments:
- x, y: Coordinate grids
- lr: Base learning rate
- number_time_steps: Number of simulation steps
- intensities: Input neuron intensity values
- rel_lr_scaling: Relative learning rate scaling for each output neuron
- n_runs: Number of simulation runs
"""
function plot_trajectories_multiple_weights(x,y,  lr, number_time_steps, intensities, rel_lr_scaling, n_runs)
    if n_runs == 1
        # Single run mode: visualize detailed evolution
        my_probs = learning_dynamics(weights_init, lr, number_time_steps, intensities, rel_lr_scaling; plotting = false)
        all_trajectories = [hcat([bary_to_cart(my_probs[n,k,:]) for n in 1:number_time_steps]...) for k in 1:K]
    else
        # Multiple runs mode: aggregate statistics
        my_probs = [learning_dynamics(weights_init, lr, number_time_steps, intensities, rel_lr_scaling; plotting = false) 
                   for _ in 1:n_runs]
    end

    if n_runs == 1
        lw = 1.2  # line width
        # Define time points to highlight
        highlight_rel_steps = [0.1, 0.25]  # 10% and 25% of total time steps
        highlight_steps = [round(Int, rel_step * number_time_steps) for rel_step in highlight_rel_steps]
        marker_styles = ["x", "o"]  # Cross and circle
        marker_size = 8

        # Plot probability evolution for each output neuron
        for k in 1:K
            figure()
            for m in 1:M
                plot(1:number_time_steps, my_probs[:,k,m], 
                     label=L"p_{%$k %$m}(k)", color=standard_colors[m], linewidth=lw)
            end
            
            ax = gca()
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=true)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.tick_params(axis="both", labelsize=standard_fontsize)
            legend(loc="center right", fontsize=standard_fontsize)
            tight_layout()
        end
        
        # Plot trajectories in probability simplex
        figure()
        z = LossLandscape(x, y; grad = false, plotting = false)
        create_standard_contour(x, y, z)
        create_standard_colorbar(z)
        my_colors = ["red", "black", "blue"]
                
        for i in 1:K
            plot(all_trajectories[i][1,:], all_trajectories[i][2,:], 
                 color=my_colors[i], label=L"\mathbf{p}_%$i(k)")
            
            # Add highlight markers for specific points on trajectories
            for (j, step) in enumerate(highlight_steps)
                plot(all_trajectories[i][1,step], all_trajectories[i][2,step], 
                     marker=marker_styles[j], color=my_colors[i], 
                     markersize=marker_size, markeredgewidth=2)
            end
        end
        
        legend(loc="upper left", fontsize=standard_fontsize)
        tight_layout()
    else
        # Multiple runs: compute distance metrics
        distances = zeros(number_time_steps,n_runs)
        for i in 1:n_runs
            for j in 1:number_time_steps
                # Compute squared Frobenius norm to identity 
                distances[j,i] = sum((my_probs[i][j,:,:] - I).^2)/2
            end
        end
        
        # Plot distance trajectories
        figure()
        for i in 1:n_runs
            plot(1:number_time_steps, distances[:,i], color="black", linewidth=0.5, alpha=0.7)
        end
        
        # Plot mean distance
        plot(1:number_time_steps, mean(distances, dims=2), color="blue", 
             linewidth=2, linestyle=":", label="Mean")
             
        legend(loc="lower left", fontsize=standard_fontsize)
        title(L"\Vert \mathbf{P}(k) - \mathbb{I}\Vert^2/2" * " with learning rates "* 
              L"10^{%$(Int(log10(lr_base)))}" * string(rel_lr_scaling), fontsize=standard_fontsize)
        xlabel("Iteration index " * L"$k$", fontsize=standard_fontsize)
        gca().ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=true)
        gca().tick_params(axis="both", labelsize=standard_fontsize)
        tight_layout()
    end
end


#===============================Helper Functions ========================================#

"""
Creates a standard colorbar for contour plots.
Arguments:
- z: The data matrix used for the contour plot
- my_fontsize: Font size for labels (default: standard_fontsize)
- num_ticks: Number of tick marks to display (default: 2)
Returns: The colorbar object
"""
function create_standard_colorbar(z, my_fontsize=standard_fontsize, num_ticks=2)
    cbar = colorbar()
    min_val = minimum(filter(!isnan, z))
    max_val = maximum(filter(!isnan, z))
    tick_values = LinRange(min_val, max_val, num_ticks)
    cbar.set_ticks(round.(tick_values, digits=2))
    cbar.set_label("Loss", fontsize=my_fontsize, labelpad=-20)
    cbar.ax.tick_params(labelsize=my_fontsize)
    return cbar
end

"""
Creates a standard contour plot with consistent formatting.
Arguments:
- x, y: Coordinate grids
- z: Data values for contour levels
- levels: Number of contour levels to display
"""
function create_standard_contour(x, y, z, levels=20)
    contourf(x', y', z', levels=levels)
    gca().set_frame_on(false)
    gca().set_xticks([])
    gca().set_yticks([])
end

# Matrix for conversion between cartesian and barycentric coordinates
A = [polygon_points[1][1] polygon_points[2][1] polygon_points[3][1]; 
     polygon_points[1][2] polygon_points[2][2] polygon_points[3][2]; 
     1. 1. 1.]
A = factorize(A)  # Pre-factorize for efficient solving

"""
Loss function defined in barycentric coordinates.
Arguments:
- p: Point in barycentric coordinates
Returns: Loss value at point p
"""
function L(p)
    #return -1/3*sum(val^3 for val in p) + 1/4 * sum(val^2 for val in p)^2
    return -1/2 * (dot(p, correlation_matrix, p))
end

"""
Gradient of the loss function in barycentric coordinates.
Arguments:
- p: Point in barycentric coordinates
Returns: Gradient vector at point p
"""
function gradientL(p)
    #return [val * (val - sum(p.^2)) for val in p]
    return  p .* (correlation_matrix * p .- dot(p, correlation_matrix, p))
end

"""
Converts from cartesian to barycentric coordinates.
Arguments:
- cart_coords: Cartesian coordinates [x, y]
Returns: Barycentric coordinates [λ₁, λ₂, λ₃]
"""
function cart_to_bary(cart_coords)
    return A \  vcat(cart_coords, 1)
end

"""
Converts from barycentric to cartesian coordinates.
Arguments:
- bary: Barycentric coordinates [λ₁, λ₂, λ₃]
- polygon_points: Triangle vertices (default: global polygon_points)
Returns: Cartesian coordinates [x, y]
"""
function bary_to_cart(bary, polygon_points=polygon_points)
    return [sum(bary .* [x[1] for x in polygon_points]), sum(bary .* [x[2] for x in polygon_points])]
end


"""
Converts a gradient vector from barycentric to Cartesian coordinates.
Arguments:
- grad_bary: Gradient vector in barycentric coordinates (3D tangent vector)
Returns: Gradient in Cartesian coordinates [dx, dy]
"""
function gradient_bary_to_cart(grad_bary)
    # Map barycentric gradient to Cartesian using the Jacobian
    # The transformation is: (x,y) = Σᵢ λᵢ * vertex_i
    # So ∂/∂λᵢ → vertex_i
    dx = polygon_points[1][1] * grad_bary[1] + 
         polygon_points[2][1] * grad_bary[2] + 
         polygon_points[3][1] * grad_bary[3]
    
    dy = polygon_points[1][2] * grad_bary[1] + 
         polygon_points[2][2] * grad_bary[2] + 
         polygon_points[3][2] * grad_bary[3]
    
    return [dx, dy]
end


#==================================Main Call====================================#

"""
Main function to run the complete analysis.
Generates visualisations for both single weight vector and multiple weight vector STDP learning.
"""
function run_analysis()
    # Single weight vector analysis
    LossLandscape(x, y; grad=true, plotting=true)
    plot_trajectories(x, y, starting_points, lr, number_time_steps)
    
    # Multiple weight vectors analysis
    plot_trajectories_multiple_weights(x, y, lr_base, number_time_steps_multiple_weights, 
                                      intensities, rel_lr_scaling, n_runs)
end

# Execute the analysis
run_analysis()
