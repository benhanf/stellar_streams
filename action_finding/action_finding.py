# This pipeline is designed to find actions for star particles around galaxies in FIREbox

# input:
# Galaxy IDs at z=0
# (finds named hdf5 files of galaxy objects containing positions, velocities, and IDs)

# output: 
# - rotated (aligned) xyz coords
# - aligned velocities
# - filter mask
# - solved actions

# CONTENTS
#1. import dependencies
#2. import files
#3. filter
#4. rotate objects
#5. fit potentials
#6. solve actions

# this pipeline was written by Benjamin Hanf 
# for questions email benhanf1@gmail.com
# last revision: March 10, 2025

#------------------

target_IDs = [11, 12, 13]

def full_action_pipeline(target_IDs):

    #1. import dependencies
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy
    import glob
    import galpy
    from galpy.actionAngle import estimateDeltaStaeckel
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.actionAngle import actionAngleIsochrone
    import galpy.potential
    from galpy.potential import IsochronePotential
    from galpy.potential import MWPotential2014 
    from astropy import units
    import agama
    import pickle
    import time
    import os
    import shutil
    
    start_time = time.time()

    #2. import files, should be named '...obj_25.hdf5' for example
    files={}
    for target_ID in target_IDs:
        for filepath in glob.iglob('/DFS-L/DATA/cosmo/jgmoren1/FIREbox/FB15N1024/objects_1200/particles_within_Rvir_object_' + str(target_ID) + '.hdf5'):
            files[str(target_ID)]=h5py.File(filepath)
            
            print("target object " + str(target_ID) + " @ " + str(filepath))
            
            dir = "./" + str(target_ID)
            
            if os.path.exists(dir):
                print("directory " + str(dir) + " already exists, overriding")
                shutil.rmtree(dir)

            os.makedirs(dir)
            
    #3. filter

    def distance_from(xcoord, ycoord, zcoord, objectx, objecty, objectz):
        dist_list = np.sqrt((xcoord - objectx)**2 + (ycoord - objecty)**2 + (zcoord - objectz)**2)
        return dist_list

    r_dict = {}
    pos_mask = {}
    inner_25_mask = {}
    h=0.6774
    min_radius = 12 / h

    for j,file in enumerate(files):
        r_dict[file] = distance_from(np.zeros(len(files[file]['stellar_x'])), np.zeros(len(files[file]['stellar_x'])), np.zeros(len(files[file]['stellar_x'])), 
                                    files[file]['stellar_x'][:] / h, files[file]['stellar_y'][:] / h, files[file]['stellar_z'][:] / h)

        pos_mask[file] = (min_radius <= r_dict[file])
        inner_25_mask[file] = ((25 / h) >= r_dict[file])
        
        # print(inner_25_mask[file])
        print(len(pos_mask[file]))
        
        mask = pos_mask[file]
        antimask = ~pos_mask[file]
            
        # fig, ax = plt.subplots()
        # plt.scatter(transformed_pos[file][0][mask] / h,transformed_pos[file][1][mask] / h, s=.1, color='royalblue')
        # plt.scatter(transformed_pos[file][0][antimask] / h,transformed_pos[file][1][antimask] / h, s=.1, color='black')
        # ax.set_xlabel('x (Mpc)')
        # ax.set_ylabel('y (Mpc)')
        # plt.title(file)
        
        # plt.savefig("./" + str(file) + "/filter_" + str(file), dpi = 300)
        
            # save mask
        # with open("./" + str(file) + "/radmask_minrad:" + str(min_radius) + '_' + str(file), 'wb') as f:
        #     pickle.dump(pos_mask[file], f)
        

    #4. rotate objects    

    #rotate star plane to align with xy
    def ang_mom_vec(pos, mass, vel):
        """
        Returns the angular momentum vector of the particles provided.
        Units are [mass]*[dist]*[vel].
        """
        angmom = (mass.reshape((len(mass), 1)) *
                np.cross(pos, vel)).sum(axis=0).view(np.ndarray)
        return angmom

    def calc_faceon_matrix(angmom_vec, up=[0.0, 1.0, 0.0]):
        """
        Returns the 'face on' matrix required to rotate the cartesian coordinates so that the z axis is 
        aligned with the angular momentum vector
        """
        vec_in = np.asarray(angmom_vec)
        vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
        vec_p1 = np.cross(up, vec_in)
        vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
        vec_p2 = np.cross(vec_in, vec_p1)

        matr = np.concatenate((vec_p1, vec_p2, vec_in)).reshape((3, 3))

        return matr

    file_pos, file_vel, faceon_matrices, L_stars_dict, transformed_pos, transformed_vel = {}, {}, {}, {}, {}, {}

    for file in files:
        
        f = files[file]

        threed_pos = np.stack((np.array(f['stellar_x']),
                            np.array(f['stellar_y']),
                            np.array(f['stellar_z'])), axis=1)
        
        pos_inner = np.stack((np.array(f['stellar_x'])[inner_25_mask[file]],
                            np.array(f['stellar_y'])[inner_25_mask[file]],
                            np.array(f['stellar_z'])[inner_25_mask[file]]), axis=1)

        threed_vel = np.stack((np.array(f['stellar_vx']),
                            np.array(f['stellar_vy']),
                            np.array(f['stellar_vz'])), axis=1)
        
        vel_inner = np.stack((np.array(f['stellar_vx'])[inner_25_mask[file]],
                            np.array(f['stellar_vy'])[inner_25_mask[file]],
                            np.array(f['stellar_vz'])[inner_25_mask[file]]), axis=1)
        
        file_pos[file]=pos_inner
        file_vel[file]=vel_inner

        L_stars_dict[file] = ang_mom_vec(file_pos[file], np.array(files[file]['stellar_mass'][inner_25_mask[file]]), file_vel[file])

        faceon_matrices[file] = calc_faceon_matrix(L_stars_dict[file], up=[0.0, 1.0, 0.0])
            
        transformed_pos[file] = np.matmul(faceon_matrices[file], threed_pos.T)
        transformed_vel[file] = np.matmul(faceon_matrices[file], threed_vel.T)
        
        print(len(threed_pos.T[0]))
            
        N=1
        size = 0.5
        
        file_pos[file]=threed_pos
        file_vel[file]=threed_vel
        
        fig, ax = plt.subplots()
        
        h=0.6774
        ax.scatter(file_pos[file].T[0][::N] / h, file_pos[file].T[2][::N] / h, s = size, label = 'original')
        ax.scatter(transformed_pos[file][0][::N] / h, transformed_pos[file][2][::N] / h, s= size, label = 'transformed')
        ax.legend(loc=1)
        ax.set_xlabel('x (Mpc)')
        ax.set_ylabel('z (Mpc)')
        
        plt.arrow(0,0,100 * L_stars_dict[file][0] / np.linalg.norm(L_stars_dict[file]), 100 * L_stars_dict[file][2] / np.linalg.norm(L_stars_dict[file]), color = 'royalblue') #calculated for entire object
        plt.arrow(0,0,0, 100 * 1, color = 'black') #aligned

        plt.savefig("./" + str(file) + "/alignment_" + str(file), dpi = 300)
        
        
        #added
        mask = pos_mask[file]
        antimask = ~mask
        fig, ax = plt.subplots()
        plt.scatter(transformed_pos[file][0][mask] / h,transformed_pos[file][1][mask] / h, s=.1, color='royalblue')
        plt.scatter(transformed_pos[file][0][antimask] / h,transformed_pos[file][1][antimask] / h, s=.1, color='black')
        ax.set_xlabel('x (Mpc)')
        ax.set_ylabel('y (Mpc)')
        plt.title(file)
        
        plt.savefig("./" + str(file) + "/filter_" + str(file), dpi = 300)
        
        # # Save
        # with open("./" + str(file) + "/position_" + str(file), 'wb') as f:
        #     pickle.dump(transformed_pos[file], f)
        

        
    #5. fit potentials

    def to_v_rho(xcoord, ycoord, zcoord, objectx, objecty, objectz, objectvx, objectvy, objectvz):
        v_rho_list = np.zeros(len(objectx))
        for i in range(len(objectx)):
        
            rho = [(-xcoord + objectx[i]), (-ycoord + objecty[i]), (0)]
            
            vel_vector = [(objectvx[i]), (objectvy[i]), (objectvz[i])]
            
            normalized_rho = rho / np.linalg.norm(rho)
            
            v_rho = np.dot(vel_vector, normalized_rho)

            v_rho_list[i] = v_rho
            
        return v_rho_list

    #reproject velocity onto galactic plane so that vr, vz, vt make basis for velocity space
    def new_to_tang(xcoord, ycoord, zcoord, objectx, objecty, objectz, objectvx, objectvy, objectvz):
        tang_list = np.zeros(len(objectx))
        for i in range(len(objectx)):
            rho = [(-xcoord + objectx[i]), (-ycoord + objecty[i]), (0)]
            
            vel_vector = [(objectvx[i]), (objectvy[i]), (objectvz[i])]
            
            normalized_rho = rho / np.linalg.norm(rho)
            
            v_rho = np.dot(vel_vector, normalized_rho)

            tang_vel = np.array(vel_vector - (v_rho * normalized_rho) - np.array([0,0,objectvz[i]])) #atan2
                                
            sign = 1
            
            ccw = np.cross([0,0,1], normalized_rho)
            
            sign = np.dot(tang_vel / np.linalg.norm(tang_vel), ccw)
            
            tang_list[i] = sign * np.linalg.norm(tang_vel)
        return tang_list

    def to_rho(x,y):
        rho_list = np.zeros(len(x))
        for i in range(len(x)):
            rho = (x[i]**2 + y[i]**2)**0.5
            rho_list[i] = rho
        return rho_list

    #fit 1- agama
    agama.setUnits(length=1, velocity=1, mass=1e10)

    pot_multipole_dict, pot_cylspline_dict = {}, {}
    v_rho_dict, vt_dict, rho_dict = {}, {}, {}

    for file in files:

        f = files[file]
        x=transformed_pos[file][0]
        y=transformed_pos[file][1]
        z=transformed_pos[file][2]
        
        vx=transformed_vel[file][0]
        vy=transformed_vel[file][1]
        vz=transformed_vel[file][2]
        
        faceon_matrix = faceon_matrices[file]
        
        v_rho_dict[file] = to_v_rho(0,0,0,x,y,z,vx,vy,vz)
        vt_dict[file] = new_to_tang(0,0,0,x,y,z,vx,vy,vz)
        rho_dict[file] = to_rho(x,y)

        #fit dm + hot gas using spherical multipole
        h=0.6774

        #calculate (mask rotated) dm positions
        dm_threed_pos = np.stack((f['dm_x'],
                            f['dm_y'],
                            f['dm_z']), axis=1)

        dm_transformed_coords = np.matmul(faceon_matrix, dm_threed_pos.T / h)
        dm_pos = dm_transformed_coords

        #dm mass array
        dm_mass = np.array(f['dm_mass']) / h 

        gas_temp = f['gas_u']
        tsel = (np.log10(gas_temp) > 3)

        gas_threed_pos = np.stack((f['gas_x'],
                            f['gas_y'],
                            f['gas_z']), axis=1)

        hot_gas_pos = np.matmul(faceon_matrix, ((gas_threed_pos)[tsel]).T)
        hot_gas_mass = np.array((f['gas_mass'])[tsel] / h)

        pos_multi = np.hstack((dm_pos, hot_gas_pos)).T
        m_multi = np.hstack((dm_mass, hot_gas_mass))

        pot_multipole_dict[file] = agama.Potential(type='Multipole', particles=(pos_multi, m_multi), symmetry='Axisymmetric', lmax=4, rmin=0.1)
        
        tsel = (np.log10(gas_temp) < 3) #heart
        cold_gas_pos = np.matmul(faceon_matrix, ((gas_threed_pos)[tsel]).T)
        cold_gas_mass = np.array((f['gas_mass'])[tsel] / h)
        
        stars_threed_pos = np.stack((f['stellar_x'],
                            f['stellar_y'],
                            f['stellar_z']), axis=1)

        star_pos = np.matmul(faceon_matrix, ((stars_threed_pos)).T)
        star_mass = np.array(f['stellar_mass']) / h

        pos_cylspline = np.hstack((star_pos, cold_gas_pos)).T
        m_cylspline = np.hstack((star_mass, cold_gas_mass))

        pot_cylspline_dict[file] = agama.Potential(type='CylSpline', particles=(pos_cylspline, m_cylspline), symmetry='Axisymmetric', mmax=4, rmin=0.1)
        
    #save fit density profile
    for file in files:
        
        pot_total = agama.Potential(pot_multipole_dict[file], pot_cylspline_dict[file])
        pot_names = ['dm + hot gas', 'stars + cold gas']

        solarRadius = 8.0  # kpc

        gridR = agama.nonuniformGrid(50, 0.01, 30.0)
        gridR00 = numpy.column_stack((gridR, gridR*0, gridR*0))    # for the radial profile at z=0
        gridz   = agama.symmetricGrid(50, 0.01, 30)
        gridR0z = numpy.column_stack((gridz*0 + solarRadius, gridz*0, gridz))  # for the vertical profile at solar radius

        grid2R, grid2z = numpy.meshgrid(gridR, gridz)  # two 2d arrays of shape (len(gridR), len(gridz))
        gridRz = numpy.column_stack((grid2R.reshape(-1), grid2z.reshape(-1)))  # array of shape (len(gridR)*len(gridz), 2)
        gridR0 = numpy.column_stack((gridR, gridR*0))  # array of shape (len(gridR), 2)  for the 1d profile

        plt.figure(figsize=(10,5))
        ax=[plt.axes([0.08,0.1,0.32,0.8]), plt.axes([0.48,0.1,0.5,0.8])]
        for i in range(len(pot_total)):
            # face-on, 1d profile
            ax[0].plot(gridR, pot_total[i].projectedDensity(gridR0, beta=0), label=pot_names[i], c=['firebrick', 'royalblue'][i])
            # edge-on, 2d contour plot
            Sigma = pot_total[i].projectedDensity(gridRz, beta=numpy.pi/2).reshape(grid2R.shape)
            ax[1].clabel(ax[1].contour(grid2R, grid2z, numpy.log10(Sigma), levels=numpy.linspace(-4.1, 0, 8),
                colors=['firebrick', 'royalblue'][i]), fmt='$10^%.0f$')
        
        ax[0].set_xlabel('R [kpc]')
        ax[0].set_ylabel(r'$\Sigma(R)$ [$M_\odot/\mathsf{kpc}^2$]')
        ax[0].set_title('face-on surface density, obj '  + str(file))
        ax[0].set_yscale('log')
        ax[0].set_ylim(1e-5,1e0)
        ax[0].legend(loc='upper right')
        ax[1].set_xlim(-0.5, 30)
        ax[1].set_xlabel('R [kpc]')
        ax[1].set_ylabel('z [kpc]')
        ax[1].set_title(r'edge-on surface density, obj ' + str(file) +  ' [$M_\odot/\mathsf{kpc}^2$]')
        None

        plt.savefig('./' + str(file) + '/aligned_surface_density_light_' + str(file), dpi = 300)
        
        pot_multipole_dict[file].export('./' + str(file) + "/pot_multipole" + '_' + str(file))
        pot_cylspline_dict[file].export("./" + str(file) + "/pot_cylspline" + '_' + str(file))
        pot_total.export("./" + str(file) + "/pot_total" + '_' + str(file))

    #6 staeckle approximation for actions

    actions_dict = {}

    for file in files:
            
        potential = agama.GalpyPotential("./" + str(file) + "/pot_total_" + str(file))
                
        x=transformed_pos[file][0][pos_mask[file]]
        y=transformed_pos[file][1][pos_mask[file]]
        z=transformed_pos[file][2][pos_mask[file]]

        vx = transformed_vel[file][0][pos_mask[file]]
        vy = transformed_vel[file][1][pos_mask[file]]
        vz = transformed_vel[file][2][pos_mask[file]]
        
        rho=rho_dict[file]
        v_rho=v_rho_dict[file]
        vt=vt_dict[file]
        
        h=0.6774
        jr,jp,jz,unbound_mask = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))

        
        print("staeckle approximation, " + str(file))
        for i in range(len(x)):
            delt = estimateDeltaStaeckel(potential,rho[i],z[i])
            aAS= actionAngleStaeckel(pot=potential,delta=delt,c=True, fixed_quad=True, interpolate=True) #c=True is the default
            try:
                jr[i],jp[i],jz[i] = aAS(rho[i]*units.kpc, v_rho[i]*units.km/units.s, vt[i]*units.km/units.s, z[i]*units.kpc, vz[i]*units.km/units.s)
                if jr[i] > 10**10:
                    unbound_mask[i] = 1
                    print(sum(unbound_mask))
                    print(jr[i], jp[i],jz[i])
            except:
                unbound_mask[i] = 1
                
        actions_dict[file] = [jr, jp, jz]
        
        # with open('./' + str(file) + '/actions_dict_' + str(file) + '.pkl', 'wb') as f:
        #     pickle.dump(actions_dict, f)


        print('finished action finding for ID ' + str(file))
        print('total actions: ' + str(len(actions_dict[file][0])))
        print("unbound percentage: " + str(sum(unbound_mask) / len(x)))

        # compile information for export

        np.savez_compressed('./' + str(file) + '/savez_arrays_' + str(file), 
                ids=files[file]['stellar_id'][pos_mask[file]],
                x=transformed_pos[file][0][pos_mask[file]], 
                y=transformed_pos[file][1][pos_mask[file]], 
                z=transformed_pos[file][2][pos_mask[file]],
                jr=actions_dict[file][0],
                jp=actions_dict[file][1],
                jz=actions_dict[file][2],
                vx=transformed_vel[file][0][pos_mask[file]],
                vy=transformed_vel[file][1][pos_mask[file]],
                vz=transformed_vel[file][2][pos_mask[file]],
                facemat=faceon_matrices[file], 
                mask=pos_mask[file])

    end_time = time.time()
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)
    print(f"time elapsed: {int(minutes)} min {seconds:.2f} sec")

full_action_pipeline(target_IDs)