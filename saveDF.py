import numpy as np
import uproot
import pandas as pd

# TPC boundaries (cm)
tpcMinX = -200
tpcMaxX = 200
tpcMinY = -200
tpcMaxY = 200
tpcMinZ = 0
tpcMaxZ = 500

# Beam center (cm)
beamCenterX = -74
beamCenterY = 0
beamCenterZ = 11000

file = uproot.open("./combined.root")
T_eval_vars = [            # variables involved with low level reconstruction and truth information
    "run",                 # run number
    "subrun",              # subrun number
    "event",               # event number
    "mc_pdg",              # PDG codes of GENIE particles
    "mc_mother",           # Track IDs of GENIE mother particles
    "mc_trackID",          # Track IDs of GENIE particles
    "mc_statusCode",       # Status codes of GENIE particles
    "mc_energy",           # Energies of GENIE particles
    "mc_vx",               # X positions of GENIE particles
    "mc_vy",               # Y positions of GENIE particles
    "mc_vz",               # Z positions of GENIE particles
    "mc_time",             # Time of GENIE particles
    "NumberDaughters",     # Number of daughters from Geant4 particles
    "pdg",                 # PDG codes of Geant4 particles
    "status",              # Status codes of Geant4 particles
    "Eng",                 # Energies of Geant4 particles
    "Mass",                # Masses of Geant4 particles
    "Px",                  # X momenta of Geant4 particles
    "Py",                  # Y momenta of Geant4 particles
    "Pz",                  # Z momenta of Geant4 particles
    "StartPointx",         # Starting X positions of Geant4 particles
    "StartPointy",         # Starting Y positions of Geant4 particles
    "StartPointz",         # Starting Z positions of Geant4 particles
    "EndPointx",           # Ending X positions of Geant4 particles
    "EndPointy",           # Ending Y positions of Geant4 particles
    "EndPointz",           # Ending Z positions of Geant4 particles
    "TrackId",             # Track IDs of Geant4 particles
    "Mother"               # Track IDs of Geant4 mother particles
]

vars = {}
vars.update(file["MyAnalyzerFilter/"]["Event"].arrays(T_eval_vars, library="np"))
for col in vars:
    vars[col] = vars[col].tolist()
wc_df = pd.DataFrame(vars)

wc_df['nu_energy'] = wc_df['Eng'].apply(lambda row: row[0]) # Energy of the neutrino in each event

wc_df['N_pi0_genie'] = wc_df.apply(  # Number of GENIE pi0 in each event
    lambda row: sum(
        x == 111 and y == 1
        for x, y in zip(row['mc_pdg'], row['mc_statusCode'])
    ),
    axis=1
)

wc_df["is_1pi0"] = wc_df["N_pi0_genie"] == 1 # Mask for 1pi0 events
wc_df["is_2pi0"] = wc_df["N_pi0_genie"] > 1  # Mask for events with more than 1 pi0

wc_df['N_ch_pion_genie'] = wc_df.apply(  # Number of charged pions in each event
    lambda row: sum(
        (x == 211 or x == -211) and y == 1
        for x, y in zip(row['mc_pdg'], row['mc_statusCode'])
    ),
    axis=1
)

wc_df['is_NC'] = wc_df.apply(  # Mask for charged current events
    lambda row: not any(x in [13, -13, 11, -11] for x in row['mc_pdg']),
    axis=1
)

def inTPC(x, y, z): # Check if a point is inside the TPC boundaries
    return (
        x >= tpcMinX and x <= tpcMaxX and
        y >= tpcMinY and y <= tpcMaxY and
        z >= tpcMinZ and z <= tpcMaxZ
    )

def distToEdge(x, y, z): # Calculate the distance from a point to the nearest edge of the TPC
    return min(abs(x-tpcMinX), abs(x-tpcMaxX), abs(y-tpcMinY), abs(y-tpcMaxY), abs(z-tpcMinZ), abs(z-tpcMaxZ))

def signedDistToEdge(x, y, z): # Calculate the distance from a point to the nearest edge of the TPC, 
    if inTPC(x, y, z):         # negative values mean outside the tpc
        return distToEdge(x, y, z)
    arr = np.array([x-tpcMinX, x-tpcMaxX, y-tpcMinY, y-tpcMaxY, z-tpcMinZ, z-tpcMaxZ])
    return arr[np.argmin(np.abs(arr))]

wc_df['in_TPC_g4'] = wc_df.apply( # Marking if Geant4 particles are inside the TPC
    lambda row: [inTPC(row['StartPointx'][x], row['StartPointy'][x], row['StartPointz'][x]) for x in range(len(row['pdg']))],
    axis=1
)

wc_df['dist_to_edge_g4'] = wc_df.apply( # Absolute distance to the nearest edge of the TPC for Geant4 particles
    lambda row: [distToEdge(row['StartPointx'][x], row['StartPointy'][x], row['StartPointz'][x]) for x in range(len(row['pdg']))],
    axis=1
)

wc_df['signed_dist_to_edge_g4'] = wc_df.apply( # Distance to the nearest edge of the TPC for Geant4 particles, negative values mean outside the TPC
    lambda row: [signedDistToEdge(row['StartPointx'][x], row['StartPointy'][x], row['StartPointz'][x]) for x in range(len(row['pdg']))],
    axis=1
)

def threshEqual(v1, v2, threshold=1e-8): # Check if two values are equal within a threshold
    return np.abs(v1-v2) <= threshold

thresh = 1e-8
genieToG4Pi0 = {}
g4ToGeniePi0 = {}
for i, row in wc_df.iterrows():
    for k, pdg in enumerate(row.pdg):
        if pdg == 111:
            for j, mc_pdg in enumerate(row.mc_pdg):
                if mc_pdg == 111 and row.mc_statusCode[j] == 1:
                    if (threshEqual(row.mc_energy[j], row.Eng[k], thresh) and
                        threshEqual(row.mc_vx[j], row.StartPointx[k], thresh) and
                        threshEqual(row.mc_vy[j], row.StartPointy[k], thresh) and
                        threshEqual(row.mc_vz[j], row.StartPointz[k], thresh)):
                        genieToG4Pi0[(i, j)] = (i, k)
                        g4ToGeniePi0[(i, k)] = (i, j)

wc_df['has_non_primary'] = wc_df.apply(  # Check if there are any cosmic or reinteraction pi0s in the event
    lambda row: any(
        pdg == 111 and (row.name, idx) not in g4ToGeniePi0
        for idx, pdg in enumerate(row['pdg'])
    ),
    axis=1
)

wc_df['N_showers'] = wc_df.apply(  # Number of EM showers in the TPC
    lambda row: sum((x in [11, -11] and inTPC(row['StartPointx'][i], row['StartPointy'][i], row['StartPointz'][i]))
                    or (x == 22 and inTPC(row['EndPointx'][i], row['EndPointy'][i], row['EndPointz'][i]))
                    for i, x in enumerate(row['pdg'])),
                    axis=1
)

wc_df['single_shower'] = wc_df['N_showers'] == 1  # Mask for events with a single EM shower

wc_df['shower_origin'] = wc_df.apply( # Identify which particles created EM showers
    lambda row: [1 if ((x in [11, -11] and inTPC(row['StartPointx'][i], row['StartPointy'][i], row['StartPointz'][i]))
                      or (x == 22 and inTPC(row['EndPointx'][i], row['EndPointy'][i], row['EndPointz'][i]))) 
                 else 0 for i, x in enumerate(row['pdg'])],
                axis = 1
)

wc_df['shower_pi0_dex'] = wc_df.apply( # Identify the index of the pi0 that created the EM shower
    lambda row: list(dict.fromkeys(
                     int(np.where(row['TrackId'] == row['Mother'][i])[0][0])
                     for i, origin in enumerate(row['shower_origin']) if origin == 1)),
    axis=1
)

wc_df['shower_position'] = wc_df.apply( # Position of the start of the EM showers
    lambda row: [(row['StartPointx'][i], row['StartPointy'][i], row['StartPointz'][i])
                 if (row['pdg'][i] in [11, -11]) and show == 1
                 else (row['EndPointx'][i], row['EndPointy'][i], row['EndPointz'][i])
                 for i, show in enumerate(row['shower_origin'])
                 if (row['pdg'][i] in [11, -11, 22]) and show == 1],
    axis=1
)

def offAxAngle(x, y, z, exact, backface): # Calculate the off-axis angles
    if exact: 
        return np.arctan2(((x-beamCenterX)**2 + (y-beamCenterY)**2)**0.5, z+beamCenterZ)*180/np.pi
    if backface:
        return np.arctan2(((x-beamCenterX)**2 + (y-beamCenterY)**2)**0.5, beamCenterZ + (tpcMaxZ-tpcMinZ))*180/np.pi
    return np.arctan2(((x-beamCenterX)**2 + (y-beamCenterY)**2)**0.5, beamCenterZ)*180/np.pi

wc_df['shower_angle_front'] = wc_df.apply( # Off-axis angle of single shower, when projected to the front face of the TPC
    lambda row: offAxAngle(row['shower_position'][0][0], row['shower_position'][0][1], row['shower_position'][0][2], False, False) if row['N_showers'] == 1 else None,
    axis = 1
)

wc_df['shower_angle_back'] = wc_df.apply( # Off-axis angle of single shower, when projected to the back face of the TPC
    lambda row: offAxAngle(row['shower_position'][0][0], row['shower_position'][0][1], row['shower_position'][0][2], False, True) if row['N_showers'] == 1 else None,
    axis = 1
)

wc_df['shower_angle_exact'] = wc_df.apply( # Off-axis angle of single shower, using the exact position of the shower
    lambda row: offAxAngle(row['shower_position'][0][0], row['shower_position'][0][1], row['shower_position'][0][2], True, False) if row['N_showers'] == 1 else None,
    axis = 1
)

wc_df['pi0_angle_front'] = wc_df.apply( # Off-axis angle of each pi0 that produced a shower, when projected to the front face of the TPC
    lambda row: [offAxAngle(row['StartPointx'][row['shower_pi0_dex'][i]], row['StartPointy'][row['shower_pi0_dex'][i]], 
                            row['StartPointz'][row['shower_pi0_dex'][i]], False, False) 
                 if row['N_showers'] > 0 else None for i in range(len(row['shower_pi0_dex']))],
    axis = 1
)

wc_df['pi0_angle_back'] = wc_df.apply( # Off-axis angle of each pi0 that produced a shower, when projected to the front face of the TPC
    lambda row: [offAxAngle(row['StartPointx'][row['shower_pi0_dex'][i]], row['StartPointy'][row['shower_pi0_dex'][i]], 
                            row['StartPointz'][row['shower_pi0_dex'][i]], False, True) 
                 if row['N_showers'] > 0 else None for i in range(len(row['shower_pi0_dex']))],
    axis = 1
)

wc_df['pi0_angle_exact'] = wc_df.apply( # Off-axis angle of each pi0 that produced a shower, when projected to the front face of the TPC
    lambda row: [offAxAngle(row['StartPointx'][row['shower_pi0_dex'][i]], row['StartPointy'][row['shower_pi0_dex'][i]], 
                            row['StartPointz'][row['shower_pi0_dex'][i]], True, False) 
                 if row['N_showers'] > 0 else None for i in range(len(row['shower_pi0_dex']))],
    axis = 1
)

wc_df.to_pickle('./data.pkl')