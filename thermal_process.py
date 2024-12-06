import matplotlib.pyplot as plt
import numpy as np

########################################################################
# @author Victor Bernal
# @author Oliver Lira

# el codigo s eira actualizando y aqui colocaremos todos los avances para
# simular procesos y ciclos termodinamicos
#######################################################################

# Constantes
R = 8.31451 
Q_comb_diesel = 43000000 #J/Kg
Kgmol_aire = 0.02897 

############################################################
#definicion de los procesos para gases ideales

# Basicamente le pones valores claves de las curvas y
# la funcion genera de vuelta  un array con V, P y T en ese 
# orden, se usaran unicamente unidades del SI y como valor default
# la constante gamma sera 1.4 (monoatomica)
############################################################

def isothermal(n, T, V_i, V_f):

    """
    Calcula la curva de isoterma para un gas ideal.
    n: numero de moles del gas
    T: temperatura en K de la isoterma
    V_i: volumen inicial del gas en m^3
    V_f: volumen final del gas en m^3
    """

    V = np.linspace(V_i, V_f, 150)
    P = (n * R * T) / V
    T = np.full_like(V, T)  
    return V, P, T

def adiabatic(n, T_i, T_f, V_i, gamma=1.4):
    """
    n: numero de moles del gas
    T_i: temperatura inicial del gas en K
    T_f: temperatura final del gas en K
    V_i: volumen inicial del gas en m^3
    gamma: factor de isoterma de adiabatica (default 1.4)

    """
    P_i = (n * R * T_i) / V_i
    Vf = V_i * (T_i / T_f)**(1 / (gamma - 1))
    V = np.linspace(V_i, Vf, 150)
    P = P_i * (V_i / V)**gamma
    T = (P * V) / (n * R)
    return V, P, T


def isometric(n,V,T_i,T_f):
    
    """
    n: numero de moles del gas
    V: volumen del gas en m^3
    T_i: temperatura inicial del gas en K
    T_f: temperatura final del gas en K
    
    """
    
    p_i = (n * R * T_i) / V
    p_f = (T_f * n * R) / V
    P = np.linspace(p_i, p_f,(150))
    T = (P*V)/(n*R)
    V = np.full_like(P, V)  
    return V, P, T

def isobar(n,P,T_i,T_f):
    
    """
    n: numero de moles del gas
    P: presion del gas en Pa
    T_i: temperatura inicial del gas en K
    T_f: temperatura final del gas en K
    """
    
    v_i = (n * R * T_i) / P
    v_f = (T_f * n * R) / P
    V = np.linspace(v_i, v_f,150)
    T = (P*V)/(n*R)
    P = np.full_like(V,P )  
    return V, P, T

############################################################|
#Ciclos ideales
############################################################

def carnot_igas(n, Vi, Vmax, T1, T2, gamma=1.4):
    """
    Simulacion del ciclo de carnot para un gas ideal y de base monoatomico
    pero se puede variar su gamma.
    
    n: numero de moles del gas
    Vi: volumen  del gas en m^3
    """
    
    # Cálculo de los puntos de volumen en el ciclo de Carnot
    V2 = Vmax * (T1 / T2)**(1 / (gamma - 1))
    V4 = Vi * (T2 / T1)**(1 / (gamma - 1))
    
    # Cálculo de procesos (presión, volumen y temperatura)
    abv, abp, T_ab = isothermal(n, T2, Vi, V2)
    bcv, bcp, T_bc = adiabatic(n, T2, T1, V2, gamma)
    cdv, cdp, T_cd = isothermal(n, T1, bcv[-1], V4)
    dav, dap, T_da = adiabatic(n, T1, T2, V4)
    
    # Cálculo de los calores de los procesos isotérmicos
    Q2 = abp[0] * abv[0] * np.log(abv[-1] / abv[0])
    Q1 = cdp[0] * cdv[0] * np.log(cdv[-1] / cdv[0])
    
    # Cálculo de la entropía en cada fase
    ab_s = np.linspace(0, Q2 / T2, len(abv))  # Isotérmico A -> B
    bc_s = np.full_like(T_bc, ab_s[-1])       # Adiabático B -> C (constante)
    cd_s = np.linspace(ab_s[-1], bc_s[-1] + (Q1 / T1), len(cdv))  # Isotérmico C -> D
    da_s = np.full_like(T_da, cd_s[-1])       # Adiabático D -> A (constante)
    
    # Gráfico 2D PV 
    plt.figure(figsize=(12, 6))
    plt.plot(abv, abp, label=rf"Isotérmica (A->B), $Q_2$ ={round(Q2)} J", color='blue')
    plt.plot(bcv, bcp, label="Adiabática (B->C)", color='red')
    plt.plot(cdv, cdp, label=rf"Isotérmica (C->D), $Q_1$ ={round(Q1)} J", color='green')
    plt.plot(dav, dap, label="Adiabática (D->A)", color='orange')
    plt.xlabel("Volumen (m³)")
    plt.ylabel("Presión (Pa)")
    plt.title("Ciclo de Carnot en Diagrama PV")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Gráfico 2D VT 
    plt.figure(figsize=(12, 6))
    plt.plot(abv, T_ab, label=rf"Isotérmica (A->B), $Q_2$ ={round(Q2)} J", color='blue')
    plt.plot(bcv, T_bc, label="Adiabática (B -> C)", color='red')
    plt.plot(cdv, T_cd, label=rf"Isotérmica (C->D), $Q_1$ ={round(Q1)} J", color='green')
    plt.plot(dav, T_da, label="Adiabática (D -> A)", color='orange')
    plt.xlabel("Volumen (m³)")
    plt.ylabel("Temperatura (K)")
    plt.title("Ciclo de Carnot en Diagrama TV")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Gráfico 2D ST 
    plt.figure(figsize=(12, 6))
    plt.plot(ab_s, T_ab, label=rf"Isotérmica (A->B), $Q_2$ ={round(Q2)} J", color='blue')
    plt.plot(bc_s, T_bc, label="Adiabática (B -> C)", color='red')
    plt.plot(cd_s, T_cd, label=rf"Isotérmica (C->D), $Q_1$ ={round(Q1)} J", color='green')
    plt.plot(da_s, T_da, label="Adiabática (D -> A)", color='orange')
    plt.xlabel("Entropía (J/K)")
    plt.ylabel("Temperatura (K)")
    plt.title("Ciclo de Carnot en Diagrama ST")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Gráfico 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(abp, T_ab, abv , label="Isotérmica (A -> B)", color='blue')
    ax.plot(bcp, T_bc, bcv, label="Adiabática (B -> C)", color='red')
    ax.plot(cdp, T_cd, cdv , label="Isotérmica (C -> D)", color='green')
    ax.plot(dap, T_da, dav, label="Adiabática (D -> A)", color='orange')
    ax.set_xlabel("Presión (Pa)")
    ax.set_ylabel("Temperatura (K)")
    ax.set_zlabel("Volumen (m³)")
    ax.set_title("Espacio PVT del Ciclo de Carnot")
    ax.legend()
    plt.show()

def otto_igas(n, Vmin, Vmax, T3, T1, gamma=1.4, cv=5/2 * R):
    # Parámetros del ciclo de Otto
    T2 = T1 * (Vmin / Vmax)**(gamma - 1)
    T4 = T3 * (Vmax / Vmin)**(gamma - 1)
    
    # Procesos del ciclo en sentido horario
    ab_v, ab_p, T_ab = adiabatic(n, T1, T2, Vmin, gamma)       # Expansión adiabática (A -> B)
    bc_v, bc_p, T_bc = isometric(n, Vmax, T2, T3)              # Reducción isocórica (B -> C)
    cd_v, cd_p, T_cd = adiabatic(n, T3, T4, Vmax, gamma)       # Compresión adiabática (C -> D)
    da_v, da_p, T_da = isometric(n, Vmin, T4, T1)              # Aumento isocórico (D -> A)
    
    # Cálculo de calores
    Q_in = n * cv * (T3 - T2)  # Calor añadido en isocórica (B -> C)
    Q_out = n * cv * (T1 - T4) # Calor liberado en isocórica (D -> A)
    
    # Gráfico 2D PV 
    plt.figure(figsize=(12, 6))
    plt.plot(ab_v, ab_p, label=f"Adiabática (A -> B)", color='darkcyan')
    plt.plot(bc_v, bc_p, label=rf"Isocórica (B -> C), $Q_{{in}}$ = {round(Q_in)} J", color='firebrick')
    plt.plot(cd_v, cd_p, label=f"Adiabática (C -> D)", color='mediumseagreen')
    plt.plot(da_v, da_p, label=rf"Isocórica (D -> A), $Q_{{out}}$ = {round(Q_out)} J", color='royalblue')
    plt.xlabel("Volumen (m³)")
    plt.ylabel("Presión (Pa)")
    plt.title("Ciclo de Otto en Diagrama PV")
    plt.legend()
    plt.grid()
    plt.show()

    # Gráfico 2D TV 
    plt.figure(figsize=(12, 6))
    plt.plot(ab_v, T_ab, label="Adiabática (A -> B)", color='darkcyan')
    plt.plot(bc_v, T_bc, label=rf"Isocórica (B -> C), $Q_{{in}}$ = {round(Q_in)} J", color='firebrick')
    plt.plot(cd_v, T_cd, label="Adiabática (C -> D)", color='mediumseagreen')
    plt.plot(da_v, T_da, label=rf"Isocórica (D -> A), $Q_{{out}}$ = {round(Q_out)} J", color='royalblue')
    plt.xlabel("Volumen (m³)")
    plt.ylabel("Temperatura (K)")
    plt.title("Ciclo de Otto en Diagrama TV")
    plt.legend()
    plt.grid()
    plt.show()

    # Gráfico 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ab_p, T_ab, ab_v, label="Adiabática (A -> B)", color='darkcyan')
    ax.plot(bc_p, T_bc, bc_v, label=rf"Isocórica (B -> C), $Q_{{in}}$ = {round(Q_in)} J", color='firebrick')
    ax.plot(cd_p, T_cd, cd_v, label="Adiabática (C -> D)", color='mediumseagreen')
    ax.plot(da_p, T_da, da_v, label=rf"Isocórica (D -> A), $Q_{{out}}$ = {round(Q_out)} J", color='royalblue')
    ax.set_xlabel("Presión (Pa)")
    ax.set_ylabel("Temperatura (K)")
    ax.set_zlabel("Volumen (m³)")
    ax.set_title("Espacio PVT del Ciclo de Otto")
    ax.legend()
    plt.show()

def diesel_igas(n,V_min,V_max,T_max,p_iso,gamma=1.4,cp=(7/2)*R,cv=(5/2) * R):
    
    T1 = (p_iso*V_min)/(n*R)
    V2 = (n*R*T_max)/p_iso
    T3 = T_max*(V2/V_max)**(gamma-1)
    T4 = T1*(((n*R*T1)/p_iso)/V_max)**(gamma-1)
    
    ab_v, ab_p, T_ab = isobar(n, p_iso, T1, T_max)                     # Expansión isobara (A -> B)
    bc_v, bc_p, T_bc = adiabatic(n, T_max, T3, V2, gamma)              # expansion adiabatica (B -> C)
    cd_v, cd_p, T_cd = isometric(n, V_max, T3,T4)                      # reduccion isometrica (C -> D)
    da_v, da_p, T_da = adiabatic(n, T4, T1, V_max, gamma)              # compresion adiabatica(D -> A)
    
    Q_in = n * cp * (T_max-T1)  # Calor añadido en isobaria (A -> B)
    Q_out = n * cv * (T4 - T3) # Calor liberado en isocórica (C -> D)
    
    
    # Gráfico 2D PV 
    plt.figure(figsize=(12, 6))
    plt.plot(ab_v, ab_p, label=f"Isobarico (A -> B)", color='darkcyan')
    plt.plot(bc_v, bc_p, label=rf"Adiabatico (B -> C), $Q_{{in}}$ = {round(Q_in)} J", color='firebrick')
    plt.plot(cd_v, cd_p, label=f"Isometrico (C -> D)", color='mediumseagreen')
    plt.plot(da_v, da_p, label=rf"Adiabatico (D -> A), $Q_{{out}}$ = {round(Q_out)} J", color='royalblue')
    plt.xlabel("Volumen (m³)")
    plt.ylabel("Presión (Pa)")
    plt.title("Ciclo de Diesel en Diagrama PV")
    plt.legend()
    plt.grid()
    plt.show()

    # Gráfico 2D TV 
    plt.figure(figsize=(12, 6))
    plt.plot(ab_v, T_ab, label="Isoobarico (A -> B)", color='darkcyan')
    plt.plot(bc_v, T_bc, label=rf"Adiabatico (B -> C), $Q_{{in}}$ = {round(Q_in)} J", color='firebrick')
    plt.plot(cd_v, T_cd, label="Isometrico (C -> D)", color='mediumseagreen')
    plt.plot(da_v, T_da, label=rf"Adiabatico (D -> A), $Q_{{out}}$ = {round(Q_out)} J", color='royalblue')
    plt.xlabel("Volumen (m³)")
    plt.ylabel("Temperatura (K)")
    plt.title("Ciclo de Diesel en Diagrama TV")
    plt.legend()
    plt.grid()
    plt.show()

    # Gráfico 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ab_p, T_ab, ab_v, label="Isobarico (A -> B)", color='darkcyan')
    ax.plot(bc_p, T_bc, bc_v, label=rf"Adiabatico (B -> C), $Q_{{in}}$ = {round(Q_in)} J", color='firebrick')
    ax.plot(cd_p, T_cd, cd_v, label="Isometrico (C -> D)", color='mediumseagreen')
    ax.plot(da_p, T_da, da_v, label=rf"Adiabatico (D -> A), $Q_{{out}}$ = {round(Q_out)} J", color='royalblue')
    ax.set_xlabel("Presión (Pa)")
    ax.set_ylabel("Temperatura (K)")
    ax.set_zlabel("Volumen (m³)")
    ax.set_title("Espacio PVT del Ciclo de Diesel")
    ax.legend()
    plt.show()

###############################################################
#ciclos reales
###############################################################


def diesel(V_cilindro = 5e-4 , P_atm = 101325,T_ext = 298 , rc = 1/16.5 ,AFR = 20,  gamma=1.4, cp=(7/2)*R, cv=(5/2) * R):
    
    """
    Los valores default son los de un motor 2.0 TDI de Volkswagen.
    
    V_cilindro: volumen del cilindro del motor
    P_atm: presión atmosférica
    T_ext: temperatura exterior
    rc: relacion de compresion, V_min/V_max
    AFR: air fuel ratio, aire/combustible
    gamma: factor adiabatigo para gas ideal
    cp: calor específico para el aire, J/(kg·K)
    cv: calor específico para el combustible, J/(kg·K)
    """
    
    
    m_aire = (V_cilindro*P_atm) / (287*T_ext)
    n_aire = (m_aire)/ Kgmol_aire
    m_diesel = m_aire / AFR
    V_min  = V_cilindro * rc
    Q_isobar = Q_comb_diesel * m_diesel
    T_b = T_ext*(1/rc)**(gamma - 1)
    T_c = (Q_isobar/(n_aire*cp)) + T_b
    P_iso = (n_aire*R*T_b)/V_min
    V_c = (n_aire*R*T_c)/P_iso
    T_d = T_c*(V_c/V_cilindro)**(gamma - 1)
    
    
    
    ######################################
    # Procesos irreversibles de admision y escape
    #####################################
    V_ir = np.linspace(V_cilindro, V_min, 150)
    P_ir = np.linspace(P_atm, P_atm, 150)
    T_ir = np.linspace(T_ext, T_b, 150)

    ######################################
    # Procesos reversibles 
    #####################################
    V_adiabat, P_adiabat, T_adiabat = adiabatic(n_aire, T_ext, T_b, V_cilindro, gamma)           # compresion adiabatica (A -> B)
    V_isobar, P_isobar, T_isobar = isobar(n_aire, P_iso, T_b, T_c)                               # Expansión isobárica   (B -> C)
    V_adiabat_2, P_adiabat_2, T_adiabat_2 = adiabatic(n_aire, T_c, T_d, V_c, gamma)              # expansión adiabática  (C -> D)
    V_isomet, P_isomet, T_isomet = isometric(n_aire, V_cilindro, T_d,T_ext)                      # reducción isométrica  (D -> A)
    
    
    #################################################################
    # Calores del ciclo térmico
    #################################################################
    Q_in = Q_isobar # Calor añadido en isobárica (B -> C)
    Q_out = n_aire * cv * (T_ext - T_d) # Calor liberado en isocórica (D -> A)

    # Cálculo del porcentaje
    porcentaje_eficiencia = (1 + (Q_in / Q_out))* 100
    print(f"Porcentaje eficiencia: {porcentaje_eficiencia:.2f}%")

    # Gráfico 2D PV 
    plt.figure(figsize=(12, 6))
    # Ahora la admisión y escape son la misma curva, al principio y al final
    plt.plot(V_ir, P_ir, label=rf"Admision y escape irreversible ", color='orange')
    plt.plot(V_adiabat, P_adiabat, label=rf"Adiabático (A -> B)", color='royalblue')
    plt.plot(V_isobar, P_isobar, label=f"Isobárico (B -> C), $Q_{{in}}$ = {round(Q_in)} J", color='darkcyan')
    plt.plot(V_adiabat_2, P_adiabat_2, label=rf"Adiabático (C -> D)", color='firebrick')
    plt.plot(V_isomet, P_isomet, label=f"Isométrico (D -> A), $Q_{{out}}$ = {round(Q_out)} J", color='mediumseagreen')
    plt.xlabel("Volumen (m³)")
    plt.ylabel("Presión (Pa)")
    plt.title("Ciclo de Diesel en Diagrama PV")
    plt.legend()
    plt.grid()
    plt.show()

    # Gráfico 2D TV 
    plt.figure(figsize=(12, 6))
    plt.plot(V_ir, T_ir, label=rf"Admision y escape irreversible ", color='orange')
    plt.plot(V_adiabat, T_adiabat, label=rf"Adiabático (A -> B)", color='royalblue')
    plt.plot(V_isobar, T_isobar, label=rf"Isobárico (B -> C), $Q_{{in}}$ = {round(Q_in)} J", color='darkcyan')
    plt.plot(V_adiabat_2, T_adiabat_2, label=rf"Adiabático (C -> D)", color='firebrick')
    plt.plot(V_isomet, T_isomet, label=rf"Isométrico (D -> A), $Q_{{out}}$ = {round(Q_out)} J", color='mediumseagreen')
    plt.xlabel("Volumen (m³)")
    plt.ylabel("Temperatura (K)")
    plt.title("Ciclo de Diesel en Diagrama TV")
    plt.legend()
    plt.grid()
    plt.show()

    # Gráfico 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P_ir, T_ir, V_ir, label=rf"Admision y escape irreversible ", color='orange')
    ax.plot(P_adiabat, T_adiabat, V_adiabat, label=rf"Adiabático (A -> B)", color='royalblue')
    ax.plot(P_isobar, T_isobar, V_isobar, label=rf"Isobárico (B -> C), $Q_{{in}}$ = {round(Q_in)} J", color='darkcyan')
    ax.plot(P_adiabat_2, T_adiabat_2, V_adiabat_2, label=rf"Adiabático (C -> D)", color='firebrick')
    ax.plot(P_isomet, T_isomet, V_isomet, label=rf"Isométrico (D -> A), $Q_{{out}}$ = {round(Q_out)} J", color='mediumseagreen')
    ax.set_xlabel("Presión (Pa)")
    ax.set_ylabel("Temperatura (K)")
    ax.set_zlabel("Volumen (m³)")
    ax.set_title("Espacio PVT del Ciclo de Diesel")
    ax.legend()
    plt.show()    

