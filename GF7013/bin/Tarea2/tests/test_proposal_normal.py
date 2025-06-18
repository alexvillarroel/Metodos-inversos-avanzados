import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

pythonpackagesfolder = '../../../..' # no modificar si no se mueve de carpeta este notebook
import sys
sys.path.append(pythonpackagesfolder)

from GF7013.probability_functions.pdf import pdf_normal
from GF7013.sampling.metropolis.proposal_normal import proposal_normal

# Número de modelos propuestos
N_models = 1e6

# Matriz de covarianza de la distribución de proposición
cov = np.array([[7, 0], [0, 4]])
rng = np.random.default_rng(42)  # Para reproducibilidad
m_0 = np.array([0, 0])  # Modelo inicial
proposal = proposal_normal(cov=cov)

models = np.zeros((2, int(N_models)))
for i in range(int(N_models)):
    # Generar un modelo aleatorio
    models[:, i] = proposal.propose(m_0)

# Covarianza de la distribución de proposición
cov_mod = np.cov(models)
# Imprimir la covarianza
print("Covarianza de la distribución de proposición:")
print(cov_mod)


# Crear la figura y los ejes 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Crear scatter 3D
N_models = models.shape[1]
scatter = ax.scatter(models[0, :], models[1, :], np.arange(N_models), 
                     c=np.arange(N_models), cmap='viridis', s=1, alpha=0.5)
ax.set_title('3D Scatter of Proposed Models (Temporal Dimension)')
ax.set_xlabel('Model Parameter 1 (a)')
ax.set_ylabel('Model Parameter 2 (θ)')
ax.set_zlabel('Proposal Order')
fig.colorbar(scatter, ax=ax, label='Model Proposal Order')
plt.show()