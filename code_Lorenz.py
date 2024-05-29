###______________ Attracteur de Lorenz ______________###

# Bibliothèques :
import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from scipy.linalg import eig
from sklearn.neighbors import NearestNeighbors

# Paramètres : 
sigma = 10
rho = 28
beta = 8 / 3

#%%
# ________________________________________________________#
# ______________ [ 1 ] Visualisation ______________ #
# ________________________________________________________#

# 1] Définir les équations de Lorenz : 
def lorenz(variables, t):
    # Variables : 
    x, y, z = variables

    # Equation de Lorenz : 
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    
    return [dxdt, dydt, dzdt]

# 2] Définir les conditions initiales des variables :
variables_initiales = [1, 1, 1]

# 3] Définir le temps : 
intervalle_temps = (0, 100)
temps = np.linspace(intervalle_temps[0], intervalle_temps[1], 10000)

# 4] Résoudre les équations de Lorenz : 
solution = solve_ivp(lambda t, y: lorenz(y, t), intervalle_temps, variables_initiales, t_eval=temps, method='RK45')
x, y, z = solution.y

# 5] Représenter graphiquement l'attracteur :  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_title("Attracteur de Lorenz")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.savefig('Lorenz_visuel')
plt.show()

#%%
# ________________________________________________________#
# ______________ [ 2 ] Vérification de la sensibilité aux CI ______________ #
# ________________________________________________________#

# 1] Trouver les points fixes : 
def points_fixes():
    points_fixes_1 = (0, 0, 0)
    points_fixes_2 = (np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1)
    points_fixes_3 = (-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1)
    
    points_fixes = [points_fixes_1, points_fixes_2, points_fixes_3]
    return points_fixes


# 2] Calculer la matrice jacobienne : 
def jacobian(x, y, z):
    J = np.array([[-sigma, sigma, 0],
                  [rho - z, -1, -x],
                  [y, x, -beta]])
    return J

# 3] Analyser la stabilité des points fixes
def stabilite(points_fixes):
    for point in points_fixes:
        x, y, z = point
        J = jacobian(x, y, z)
        valeurs_propres = eig(J)[0]
        if np.all(np.real(valeurs_propres) < 0):
            stabilite = "Stable"
        else:
            stabilite = "Instable"
        print("Point fixe", point)
        print("Valeurs propres:",valeurs_propres)
        print("Stabilité:", stabilite)
        print("")

# 4] Appeler les fonctions :
points_fixes = points_fixes()
print("\nAnalyse de stabilité:")
print("")
stabilite(points_fixes)

#%%
# ________________________________________________________#
# ______________ [ 3 ] Indépendence des conditions initiales ______________ #
# ________________________________________________________#

# 1] Choisir les conditions initiales à comparer : 
condition_initiale_a = [1, 1, 1]
condition_initiale_b = [1, -1, -1]

# 2] Définir le temps de la simulation : 
intervalle_temps_CI = (0, 200) 
temps_CI = np.linspace(intervalle_temps_CI[0], intervalle_temps_CI[1], 20000) 

# 3] Résoudre le équations de Lorenz : 
solution_a = solve_ivp(lambda t, y: lorenz(y, t), intervalle_temps_CI, condition_initiale_a, t_eval=temps_CI)
solution_b = solve_ivp(lambda t, y: lorenz(y, t), intervalle_temps_CI, condition_initiale_b, t_eval=temps_CI)

# 4] Extraire les valeurs de x : 
valeurs_x_a = solution_a.y[0]
valeurs_x_b = solution_b.y[0]

# 5] Tracer les histogrammes :
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(valeurs_x_a, bins=60, density=True, color='red')
plt.title('Histogramme de x (condition initiale [1, 1, 1])')
plt.xlabel('x')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(valeurs_x_b, bins=60, density=True, color='red')
plt.title('Histogramme de x (condition initiale [1, -1, -1])')
plt.xlabel('x')
plt.ylabel('Density')

plt.tight_layout()
plt.savefig('histogramme')
plt.show()

#%%
# ________________________________________________________#
# ______________ [ 4 ] Détermination la dimension d’un attracteur ______________ #
# ________________________________________________________#

# 1] Calculer la dimension :
def dimension_locale(points, k=10):
    pp_voisins = NearestNeighbors(n_neighbors=k+1).fit(points)
    distances_voisins, indices_voisins_voisins = pp_voisins.kneighbors(points)
    dimension = []
    for i in range(len(points)):
        r = distances_voisins[i, 1:] # distance des k voisins les plus proches 
        C_r = np.arange(1, k+1) # comptage cumulatif des voisins dans le rayon r 
        log_r = np.log(r) 
        log_C_r = np.log(C_r)
        p = np.polyfit(log_r, log_C_r, 1) # pente 
        local_dim = p[0]
        dimension.append(local_dim)
    return np.array(dimension)

# 2] Appliquer la fonction : 
t = np.linspace(0, 100, 10000)
etat_initial = [1, 1, 1]  
points = odeint(lorenz, etat_initial, t)
dimentions_locales = dimension_locale(points, k=10)

# 3] Calculer la moyenne des dimensions locaux :
dimention_locale_moyenne = np.mean(dimentions_locales)
print("Dimension local moyenne :", dimention_locale_moyenne)

# 4] Représentation graphique : 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(points[:,0], points[:,1], points[:,2], c=dimentions_locales, cmap='viridis', s=1)
plt.colorbar(sc, label='dimension locale')
ax.set_title("Attracteur de Lorenz")
plt.savefig('Lorenz_dimension')
plt.show()