"""
This module plots a representation of firing maps.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from .PlotHelper import add_goal_zone, add_obstacles



class RepresentationVisualization():
    """ Plotting a representation of firing maps.
    """

    def __init__(self, data_path, flname, fig_path, cal_covarage, firing_rate_quota, param_file, resolution=20):
        self.read_sim_data(data_path='../openfield', sim_file=param_file['sim'])
        with open(os.path.join(data_path, param_file['net'])) as fl:
            self.nps = json.load(fl)
        with open(os.path.join(data_path, flname)) as fl:
            self.cell_centers = np.array(json.load(fl))
        self.Field_data_filnam = 'coverage_summary.dat'
        self.max_fr_grids = max(self.cell_centers[:, 5])
        self.fig_path = fig_path
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        self.fig_path_sub = os.path.join(fig_path, "individual-fields")
        if not os.path.exists(self.fig_path_sub):
            os.makedirs(self.fig_path_sub)
        self.cal_covarage = cal_covarage
        fig_margin = self.assign_fig_margin()
        self.resolution = resolution #TODO: Check default resolution value
        self.x_pos = np.linspace(self.cell_centers[:, 0].min() - fig_margin,
                                 self.cell_centers[:, 0].max() + fig_margin, resolution)
        self.y_pos = np.linspace(self.cell_centers[:, 1].min() - fig_margin,
                                 self.cell_centers[:, 1].max() + fig_margin, resolution)
        ################
        #        self.x_pos = np.linspace(0,0.5, 50)
        #        self.y_pos = np.linspace(0,0.5, 50)
        ################
        self.firing_rate_quota = firing_rate_quota

    def assign_fig_margin(self):
        if self.cal_covarage:
            return 0
        else:
            return 0.2

    def read_sim_data(self, data_path, sim_file):
        """ This function reads the simulation parameters in the json file.

        Parameters
        ----------
        data_path : string
            path to the json file.
        sim_file : string
            file name

        Returns
        -------
        None.

        """
        self.sim_file_path = os.path.join(data_path, sim_file)
        with open(self.sim_file_path, 'r') as fl:
            net_dict = json.load(fl)
        self.data_path = net_dict["data_path"]
        self.sim_dict = net_dict
        self.sim_env = net_dict['sim_env']
        self.env_limit_dic = net_dict['environment'][self.sim_env]
        if self.sim_env == 'openfield':
            self.xmin_position = float(self.env_limit_dic['xmin_position'])
            self.xmax_position = float(self.env_limit_dic['xmax_position'])
            self.ymin_position = float(self.env_limit_dic['ymin_position'])
            self.ymax_position = float(self.env_limit_dic['ymax_position'])
            self.opn_fld_xlim = np.array([self.xmin_position, self.xmax_position])
            self.opn_fld_ylim = np.array([self.ymin_position, self.ymax_position])
            self.hide_goal = net_dict['goal']['hide_goal']
            self.reward_recep_field = net_dict['goal']['reward_recep_field']
            self.goal_x = net_dict['goal']['x_position']
            self.goal_y = net_dict['goal']['y_position']
            # Used to rewrite the environment limits to start from 0
            self.h_label = [0, self.opn_fld_xlim[1] - self.opn_fld_xlim[0]]
            self.v_label = [0, self.opn_fld_ylim[1] - self.opn_fld_ylim[0]]
        else:
            print("environment {} undefined".format(self.sim_env))
            

        self.opn_fld_xlim = np.array([self.xmin_position, self.xmax_position])
        self.opn_fld_ylim = np.array([self.ymin_position, self.ymax_position])
        self.hide_goal = net_dict['goal']['hide_goal']
        self.reward_recep_field = net_dict['goal']['reward_recep_field']
        self.goal_x = net_dict['goal']['x_position']
        self.goal_y = net_dict['goal']['y_position']

    def compute_grid_firingfield(self, pars):
        """ returns the firing field of a single grid cell.
        Parameters
        ----------
        pars : list
            pars[0]: horizontal offset of the grid
            pars[1]: vertical offset of the grid
            pars[2]: Lambda parameter
            pars[3]: kappa parameter
            pars[4]: place=0, grid=1, border=2
            pars[5]: max firing rate

        Returns
        -------
        field : np.array
            firing field of a single place cell.

        """
        i_ = np.arange(0, 3)
        field = np.zeros((self.x_pos.size, self.y_pos.size))
        for xi, x in enumerate(self.x_pos):
            for yi, y in enumerate(self.y_pos):
                X = np.array([x, y])
                X0 = np.array([pars[0], pars[1]])
                omega = 2 * np.pi / (np.sin(np.pi / 3) * pars[3])
                phil = -np.pi / 6 + i_ * np.pi / 3
                kl = np.array([np.cos(phil), np.sin(phil)]).T
                point_fr = pars[5] * np.exp(pars[2] * (np.sum(np.cos(omega * np.dot(kl, X - X0))) - 3) / 3)
                field[yi, xi] = point_fr
        return field

    def compute_place_firingfield(self, pars):
        """ returns the firing field of a single place cell.
        Parameters
        ----------
        pars : list
            pars[0]: the x coordinate of the place cell
            pars[1]: the y coordinate of the place cell
            pars[2]: standard deviation (horizontal)
            pars[3]: standard devitaion (vertical)
            pars[4]: indicates type of the cell
            pars[5]: max firing rate
        Returns
        -------
        field : np.array
            firing field of a single place cell.

        """
        field = np.zeros((self.x_pos.size, self.y_pos.size))
        for xi, x in enumerate(self.x_pos):
            for yi, y in enumerate(self.y_pos):
                i = ((x - pars[0]) / pars[2]) ** 2
                j = ((y - pars[1]) / pars[3]) ** 2
                field[xi, yi] = np.exp(-(i + j) / 2) * pars[5]
        return field

    def compute_border_firingfield(self, pars):
        """ returns the firing field of a single border cell.
        Parameters
        ----------
        pars : list
            pars[0]: horizontal offset of the grid
            pars[1]: vertical offset of the grid
            pars[2]: standard deviation (horizontal)
            pars[3]: standard devitaion (vertical)
            pars[4]: indicates type of the cell
            pars[5]: max firing rate
        Returns
        -------
        field : np.array
            firing field of a single border cell.

        """

        field = np.zeros((self.x_pos.size, self.y_pos.size))
        for xi, x in enumerate(self.x_pos):
            for yi, y in enumerate(self.y_pos):
                if pars[0] - pars[2] <= x and x <= pars[0] + pars[2]:
                    if pars[1] - pars[3] <= y and y <= pars[1] + pars[3]:
                        # x and y are swapped here due to the way that pcolor works, see matplotlib page for details
                        field[yi, xi] = pars[5]  
        return field

    def write_header(self):
        """ Writing the header of the performance data file.
        """
        self.log_file.write('{}\t{}\t{}\t{}\t{}\t'.format('p_nrows',
                                                          'p_ncols',
                                                          'p_row_sig',
                                                          'p_col_sig',
                                                          'p_fr'))
        self.log_file.write('{}\t{}\t{}\t{}\t{}\t'.format('g0_nrows',
                                                          'g0_ncols',
                                                          'g0_kap',
                                                          'g0_lam',
                                                          'g0_fr'))
        # self.log_file.write('{}\t{}\t{}\t{}\t{}\t'.format('g1_nrows',
        #                                                   'g1_ncols',
        #                                                   'g1_kap',
        #                                                   'g1_lam',
        #                                                   'g1_fr'))
        # self.log_file.write('{}\t{}\t{}\t{}\t{}\t'.format('g2_nrows',
        #                                                   'g2_ncols',
        #                                                   'g2_kap',
        #                                                   'g2_lam',
        #                                                   'g2_fr'))
        # self.log_file.write('{}\t{}\t{}\t{}\t{}\t'.format('g3_nrows',
        #                                                   'g3_ncols',
        #                                                   'g3_kap',
        #                                                   'g3_lam',
        #                                                   'g3_fr'))
        self.log_file.write('{}\t{}\t'.format('average_coverage',
                                              'firing_rate_quota'))

        self.log_file.write('\n')

    def save_coverage_log(self):
        ############
        place_params = self.nps['place']['cells_prop']
        grid_params = self.nps['grid']['cells_prop']
        self.log_file.write('{}\t{}\t{}\t{}\t{}\t'.format(place_params['p_nrows'],
                                                          place_params['p_ncols'],
                                                          place_params['p_row_sigma'],
                                                          place_params['p_col_sigma'],
                                                          place_params['max_fr']))
        self.log_file.write('{}\t{}\t{}\t{}\t{}\t'.format(grid_params['g_nrows'][0],
                                                          grid_params['g_ncols'][0],
                                                          grid_params['g_kappa'][0],
                                                          grid_params['g_lambda'][0],
                                                          grid_params['max_fr'][0]))
        # self.log_file.write('{}\t{}\t{}\t{}\t{}\t'.format(grid_params['g_nrows'][1],
        #                                                   grid_params['g_ncols'][1],
        #                                                   grid_params['g_kappa'][1],
        #                                                   grid_params['g_lambda'][1],
        #                                                   grid_params['max_fr'][1])),
        self.log_file.write('{}\t{}'.format(self.mean_coverage,
                                            self.firing_rate_quota))
        self.log_file.write('\n')

    ###############

    def calculate_covarage(self, center, field):
        mean_ind_field = np.min(field)
        #        mean_ind_field = np.round(mean_ind_field)
        env_fr_thold = (self.firing_rate_quota) * center[
            5]  # mean_ind_field + (center[5]-mean_ind_field)*(1-self.firing_rate_quota)
        coverage_rate = np.count_nonzero(field > env_fr_thold) / np.prod(field.shape)
        field = np.where(field < env_fr_thold, 0, field)
        return field, coverage_rate

    def plot_firing_maps(self, frm='pdf', save=True,
                         plot_inds=True, pop_up=False, show_obs=False):
        """
        plots sum of the firing maps of place and grid cells:
            - colorbar in range of 0 to max(fr)
            - colorbar in range of min(fr) to max(fr)
        Parameters
        ----------
        frm : format of the fure
            indicades the format of the figure
        save : bool
            the figure would be saved
        plot_inds : TYPE, optional
            DESCRIPTION. The default is True.
        pop_up : bool
            the figure pops up

        Returns
        -------
        None.

        """
        
        list_coverage = []
        num_place = 0
        num_grid = 0
        sum_field = np.zeros((self.x_pos.size, self.y_pos.size))
        fig_filepath = os.path.join(self.fig_path, "firing-field")
        fig_filepath_sub = os.path.join(self.fig_path_sub, "firing-field")
        total_tl_list = []
        for i, center in enumerate(self.cell_centers):
            sys.stdout.write('\rvisualizing firing maps {} of {} spatial '
                             'selective cells'.format(i, len(self.cell_centers)))
            sys.stdout.flush()
            if center[4] == 0:
                field = self.compute_place_firingfield(center)
                ind_tl = r'$N_{PC} = $'+'{}, '.format(num_place) 
                cell_params = r"$\sigma_x={}, \sigma_y={}$".format(center[3], center[2])
                num_place += 1
                if cell_params not in total_tl_list:
                    total_tl_list.append(cell_params)
                ind_tl += cell_params
            elif center[4] == 1:
                field = self.compute_grid_firingfield(center)
                ind_tl = r'$N_{GC} = $'+'{}, '.format(num_grid) 
                cell_params = r"$\lambda={}, \kappa={}$".format(center[3], center[2])
                num_grid += 1
                if cell_params not in total_tl_list:
                    total_tl_list.append(cell_params)
                ind_tl += cell_params
            elif center[4] == 2:
                field = self.compute_border_firingfield(center)
                ind_tl = "delta_x={}, delta_y={}, {}".format(center[3] * 2, center[2] * 2, 8)
            elif center[4] == 3:
                field = self.compute_border_firingfield(center)
                ind_tl = "delta_x={}, delta_y={}, {} - obstacle cell".format(center[3] * 2, center[2] * 2, 8)
            if self.cal_covarage and center[4] != 2:
                field, covarage_rate = self.calculate_covarage(center, field)
#                coverage_text = '\n firing range quota = {}, field coverage={}'.format(
#                    np.round(1 - self.firing_rate_quota, 2), covarage_rate)
                list_coverage.append(covarage_rate)
            if center[4] in [0, 1]:
                sum_field += field

            start_point = [self.sim_dict['start']['x_position'],
                           self.sim_dict['start']['y_position']]

            if plot_inds:  # saves individual fields
                fig, ax = plt.subplots()
                im = ax.pcolor(self.x_pos,
                               self.y_pos,
                               field,
                               edgecolors='none',
                               vmin=0,
                               shading='auto')
                ax.set_ylabel('y')
                ax.set_xlabel('x')
                plt.scatter(start_point[0],
                            start_point[1],
                            marker='X', color='r')
                if not self.sim_dict['goal']['hide_goal']:
                    add_goal_zone(ax, self.sim_dict)
                ax.set_title(ind_tl)

                ax.hlines(self.opn_fld_ylim[0], self.opn_fld_xlim[0], self.opn_fld_xlim[1],
                          color='black', label='environment')
                ax.hlines(self.opn_fld_ylim[1], self.opn_fld_xlim[0], self.opn_fld_xlim[1],
                          color='black')
                ax.vlines(self.opn_fld_xlim[0], self.opn_fld_ylim[0], self.opn_fld_ylim[1],
                          color='black')
                ax.vlines(self.opn_fld_xlim[1], self.opn_fld_ylim[0], self.opn_fld_ylim[1],
                          color='black')
                
                # rewrite the environment limits to start from 0
                plt.xticks(ticks=self.opn_fld_xlim, labels=self.h_label)
                plt.yticks(ticks=self.opn_fld_ylim, labels=self.v_label)
                
                if show_obs:
                    add_obstacles(ax, self.sim_dict)
                
                # ax.set_xlim([0, 5000])
                plt.colorbar(im, label = "Firing Rate (Hz)")
                fig.savefig(fig_filepath_sub + '-unit{}.{}'.format(i, frm), format=frm)
                plt.close(fig)
        total_tl = str(total_tl_list)
        total_tl = total_tl.replace("['", "")
        total_tl = total_tl.replace("']", "")
        total_tl = total_tl.replace("', '", "\n")
        total_tl = total_tl.replace('["', "")
        total_tl = total_tl.replace('"]', "")
        total_tl = total_tl.replace('", "', "\n")
        total_tl = total_tl.replace(r'\\', "\\")
        
        num_cells_text = ''
        if num_place:
            num_cells_text += ("$N_{PC} = $" + f"{num_place}")
            if num_grid:
                num_cells_text += r", $N_{GC} = $" + f"{num_grid}"
        elif num_grid:
            num_cells_text += (r"$N_{GC} = $" + f"{num_grid}")
        else:
            raise ValueError # Why would you want to run this with no cell populations to visualize?
        num_cells_text += '\n'
            
        #num_cells_text = '\n' + r"$N_{GC} = $" + f"{num_grid}, " + "$N_{PC} = $" + f"{num_place}"


        fig_options = [(None, ""), (0, "_0_")]
        for item in fig_options:
            fig, ax = plt.subplots(1, 1, figsize=(9, 7.25))  # the color bar range starts always from min value
            ax.set_aspect(1)
            im = ax.pcolor(self.x_pos,
                           self.y_pos,
                           sum_field,
                           edgecolors='none',
                           shading='auto',
                           vmin=item[0])
            ax.scatter(start_point[0],
                       start_point[1],
                       marker='X',
                       color='r')
            #ax.set_ylabel('y (a.u.)')
            #ax.set_xlabel('x (a.u.)')
            if self.sim_dict['goal']['hide_goal'] == False:
                add_goal_zone(ax, self.sim_dict)
                
            if self.nps['border']['num_neurons']:
                env_col = 'yellow'
                env_lw = 3
            else:
                env_col = 'black'
                env_lw = 1
            ax.hlines(self.opn_fld_ylim[0], self.opn_fld_xlim[0], self.opn_fld_xlim[1],
                      color=env_col, lw=env_lw, label='environment')
            ax.hlines(self.opn_fld_ylim[1], self.opn_fld_xlim[0], self.opn_fld_xlim[1],
                      color=env_col, lw=env_lw)
            ax.vlines(self.opn_fld_xlim[0], self.opn_fld_ylim[0], self.opn_fld_ylim[1],
                      color=env_col, lw=env_lw)
            ax.vlines(self.opn_fld_xlim[1], self.opn_fld_ylim[0], self.opn_fld_ylim[1],
                      color=env_col, lw=env_lw)
            ax.set_title(num_cells_text + total_tl)
            plt.colorbar(im, label = "Firing Rate (Hz)")
            # rewrite the environment limits to start from 0
            plt.xticks(ticks=self.opn_fld_xlim, labels=self.h_label)
            plt.yticks(ticks=self.opn_fld_ylim, labels=self.v_label)
            if pop_up:
                plt.show()
                plt.close(fig)
            else:
                pass
            if save:
                fig.savefig(fig_filepath + item[1] + ".{}".format(frm), format=frm)
            else:
                pass
            if pop_up:
                plt.show()
            else:
                pass
        if self.cal_covarage:
            reset_summary_file = False
            field_data_path = os.path.join(*self.data_path.split('/')[:-2])
            field_data_path = os.path.join(field_data_path, self.Field_data_filnam)
            print('\n writing summary of coverage in {}'.format(field_data_path))
            file_exists = os.path.exists(field_data_path)
            if file_exists and reset_summary_file:
                os.remove(field_data_path)
                file_exists = False
            self.log_file = open(field_data_path, 'a')  # creates the text file
            if not file_exists:
                self.write_header()
#            self.save_coverage_log()
