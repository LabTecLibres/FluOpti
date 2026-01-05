# -*- coding: utf-8 -*-
"""
Analysis functions developed for the FluoPi/FluOpti proyect.

@author: Prosimios
"""
import matplotlib as mpl

# to use the output figures in vectorial softwares
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import cv2
import math

# import glob
import os
import re  # to extract numbers from strings
import pickle as pkl
import inspect  # to manage **kwargs
from time import time as ttime  # ,localtime

# from time import localtime

from skimage.transform import rescale
from skimage.filters import gaussian
from skimage.metrics import structural_similarity as ssim  # to eval smoothing
import skimage.feature as skfeat
from skimage.morphology import disk

from scipy.optimize import curve_fit
from scipy import signal

from PIL import Image as pil_image
from io import BytesIO

# to use interactive widgets
import ipywidgets as widgets
from IPython.display import display, clear_output

# import fluopi functions
try:
    import plotting as flup
except:
    import fluopi.plotting as flup

###################
## General lists ##

# Define the image default channels dimentional position and name
CHANNELS = {0: "R", 1: "G", 2: "B"}
# structured as {channel_position : channel_name}
# --> channels_positions = list(CHANNELS.keys())

# default color to be used in plots for each channel position
COLORS = {0: "r", 1: "g", 2: "b"}

###############################
####### Objet Classes #########


class DataSet:
    """
    DataSet class objects contains a specific data set (e.g. white light data)
    with its associated time vector and file names.
    It also contains methods to diplay its data and its information.
    """

    def __init__(self, name):
        self.name = name  # string
        self.data = dict()  # dictionary
        self.times = dict()  # dictionary of numeric list or array
        self.fnames = dict()  # dictionary of list
        self.channels = (
            list()
        )  # list ['W','F'][Whithe_light, Blue_light], the keys of the othe dictionaries
        self.rois = dict()  # dict with rois
        self.plot_path = ""  # path of plots folder
        self.px_to_mm = None  # pixel to millimeter convertion factor
        self.smoothing = dict()  # to store the smoothing parameters
        self.background = dict()  # to store the channels background
        self.threshold = 255  # whithe field threshold to binarize images
        self.band_params = dict()  # to store the parameters of border band definition
        self.shift = {"y": None, "x": None}  # shift spline in each axis
        self.ref_center = [
            None,
            None,
        ]  # [y_position, x_positon] of reference element center

    # Methods
    def add_data(self, data, fnames):
        """
        To add some data to dataset

        data: dictionary with the data
        fnames: list with the file names

        It gets the channels list directly fromt data.keys()
        """

        self.data = data
        self.fnames = fnames
        self.channels = list(data.keys())
        return "Data added succesfully"

    def get_times(self, param_name="time", file_extension="txt"):
        """
        to get the time information from metadata files asociated to the
        filenames in the data set.
        Use them to construct the propper time vector.
        """
        # self.times =
        return

    def compose_image(self, channels, data_frame=-1, im_structure="RGB", dtype="uint8"):
        """
        channels: dict
            dictionary with the channels positions and names to built the image
            e.g. channels = { 0 : 'R', 1 : 'G'}, indicate the image will be composed
            of with the channels R and G of the data in positions 0 and 1
            of the new image respectivelly.

        data_frame is not necessary for channels with just 1 frame

        im_structure: string
            'RGB' or 'RGBA'

        """
        # get the channels positions
        c_positions = list(channels.keys())

        # get the data
        data = self.data

        # get actual data channels
        dchans = list(data.keys())

        # check if input channels are in data channels
        checked_pos = []
        for pos in c_positions:

            chan = channels[pos]

            if chan not in dchans:
                print(
                    f"Channel {chan} cannot be found in dataset. It will be added as zeros."
                )
            else:
                checked_pos.append(pos)

        # create the array to rebuilt the image based on the first checked channel
        n, m = data[checked_pos[0]].shape[0:2]  # take just the first two dimentions

        if im_structure == "RGBA":
            imdata = np.zeros((n, m, 4), dtype=dtype)

        else:
            imdata = np.zeros((n, m, 3), dtype=dtype)

        # compose an np.array of the image based on the verified channels
        for i in checked_pos:

            chan = channels[i]

            data_chani = data[chan]

            if data_chani.ndim == 3:
                try:
                    imdata[:, :, i] = data_chani[:, :, data_frame]
                except:
                    print(f"frame {data_frame} cannot be found for channel '{chan}'")

            elif data_chani.ndim == 2:
                imdata[:, :, i] = data[chan][:, :]

        return imdata

    def plot_frame_channels(self, frame, channels=None):
        """
        Shows the desired image frame channels from the data

        Parameters
        ----------

        frame : int
            number in dataS of the image frame to be shown

        channels : list, optional
            list of channels to be shown, by default None

        Returns
        -------
        None, just shows the image
        """

        if channels is None:
            channels = self.channels

        nchans = len(channels)

        data = self.data

        fig, axs = plt.subplots(
            1, nchans, figsize=(4 * (nchans), 4), layout="constrained"
        )
        fig.suptitle(f"Image {frame=}", fontsize=12)

        if nchans == 1:
            axs = [axs]

        for i in range(nchans):
            chani = channels[i]

            imi = axs[i].imshow(data[chani][:, :, frame])
            fig.colorbar(imi, fraction=0.035)
            axs[i].set_title(f"{chani} channel")

        plt.show()

    def display_chan_info(self, chan):
        """
        display some information of indicated input channel
        """
        print(f"Data shape: {self.data[chan].shape}")
        print(f"Data type: {self.data[chan].dtype}")

    def display_control_regime(self, **kwargs):
        """
        Plot the dataset control regime

        Parameters
        ----------

        kwargs:
            see flup.display_control_regime parameters

        """
        flup.display_control_regime(self, **kwargs)

    def __str__(self):
        return f"Dataset {self.nombre}"


class ROI:
    """
    ROI (Region Of Interest) element asociated to a colony surrounding area.
    ROIs belong to a DataSet.rois
    It also contains methods to diplay its data and its information.
    """

    def __init__(
        self,
        col_id,
        data,
        xlims,
        ylims,
        blob,
        center=None,
        chans_times=dict(),
        descriptors=dict(),
    ):

        self.id = col_id  # integer
        self.data = data  # dictionary with data arrays of each channel

        # limits coordinates in the source image (they are the actuall limits, need to add +1 to perform slicing)
        self.xlims = xlims  # [x_left, x_right]
        self.ylims = ylims  # [y_up, y_dowm]
        self.blob = blob  # [yc, xc, rc], rc is the blob colony radius and yc,xc are the center coordinates in the source image.

        self.chans_times = chans_times  # {channel_name: time_serie_name} to map each channel to a serie in self.times
        self.descriptors = descriptors  # {channel_name: descriptive_text}
        self.times = dict()  # {time_serie_name: time_values}

        ###  computed values  ###
        self.hw = self.get_shape()  # [heigh, wide] of the roi

        # colony center coordinates [yc, xc] relative to ROI coordinates
        if center is None:
            self.center = (
                self.get_center()
            )  # use the roi central point as a first guess
        else:
            self.center = center  # [yc,xc] colony center coordinates in ROI

        # colony center coordinates [yc, xc] relative to source image coordinates

        self.rroi = (
            0.5 * (np.asarray(self.hw) - 1).mean()
        )  # roi radius taken as half the mean of the heigh and wide sizes
        self.nframes = self.get_nframes()
        self.channels = list(self.data.keys())  # list of channels names

        self.rms = None  # init the Radial Mean Signal attribute

    # Methods
    def get_shape(self, chan=None):
        """
        To get the roi data height and wide shape, based on the data of the
        idicated channel. I no channel is indicated, use the first one.
        """
        if chan is None:
            chan = list(self.data.keys())[0]

        h, w = self.data[chan].shape[0:2]

        return (h, w)

    def get_center(self, rounded=False):
        # get the central point of the ROI

        h, w = self.hw

        # geometric center
        center = np.asarray([h / 2, w / 2])  # maybe round them to make integers?

        if rounded:
            center = round_up(center)

        return center

    def get_source_center(self, center=None, rounded=False):
        """
        get the position of the colony central point [yc, xc] relative to the source image coordinates

        Parametes:
        ---------
        center: list
            [yc, xc] center position values relative to the ROI coordinates
            By default it is used the "representative colony center" (typically
            obtained from the last frame).
            But this function can be used for custom center values (e.g. to get
            the source center using exact colony center value in each frame)

        rounded: bool
            If True, the returned values are rounded to the nearest integer with
            the round_up() function.
        """

        # use the colony "representative center" by default
        if center == None:
            center = self.center

        # compute the values
        source_cy = self.ylims[0] + center[0]
        source_cx = self.xlims[0] + center[1]

        source_center = np.asarray([source_cy, source_cx])

        if rounded:
            source_center = round_up(source_center)

        return source_center

    def get_nframes(self, channel=None):

        if channel is None:
            channel = list(self.data.keys())[0]

        dc = self.data[channel]

        if dc.ndim == 2:
            dc = np.expand_dims(dc, axis=-1)  # add 3rd dimmention

        nframes = dc.shape[2]

        return nframes

    def update_chans_times(self, chans_times):
        """
        update the channels to times relation based in the given chans_times
        dictionary. It adds the keys elements, replace the key elements in self
        which are in the new chans_times, and keeps the key elements which are
        only in self.
        """
        self.chans_times.update(chans_times)

    def update_channels(self, chans_times={}, descriptors=None):
        """
        chans_times: dict
            to map channels to a time serie stores in self.times
            e.g. {'chan1': 'times3', 'chan2': 'times1', 'chan4': 'times1'}
            channels not included in the dict will keep its previous values
            or None of there isn't a previous one.

        descriptors:dict
            are text to detail the channel content.
            dictionary of structure:
                {channel : 'descriptor text'}
        """

        # update the channel name list based on the self.data
        self.channels = list(self.data.keys())

        # update the descriptors
        if descriptors is not None:

            for chan in descriptors.keys():

                # verify its an actual ROI channel
                if chan in self.channels:

                    self.descriptors[chan] = descriptors[chan]

        # update the channels to time point series relation
        ctkeys = list(chans_times.keys())

        # go for each roi channel
        for roichan in self.channels:

            # if it is present on the input list, update its relation
            if roichan in ctkeys:
                self.chans_times[roichan] = chans_times[roichan]

            # if its not present in the self roi relation, assign None
            if roichan not in self.chans_times.keys():
                self.chans_times[roichan] = None

    def add_times(self, tkey, times=None, dt=1, t0=0):
        """
        Not sure if make it a dictionary to store diffetent times from different
        metadatas, or use just one of them as aproximation...
        """
        if times is not None:

            self.times[tkey] = times

        else:

            nframes = self.nframes

            self.times[tkey] = np.arange(t0, nframes, dt)

    def check_channels(self, new_channels, ask=True, display=False):
        """
        to check if the new channels are currently in the ROI and
        to avoid replace a channel accidentally you can ask user confirmation

        Parameters
        ----------
        new_channels : list
            new channel names to be adde to the roi

        ask: bool
            True to ask user confirmation in case of coinceidence.
            False to return directly the result

        display: bool
            if True, display informative messages in case foun the channels.

        Returns
        -------
        True: indicates one of the new channels is currently in the ROI or
            the user stop the replace

        False: any of the new_channels was present in the roi.channels or
            the user indicates to replace them.

        """
        # update the roi channels
        self.update_channels()

        # check the new channels are not previously stored
        for new_chan in new_channels:

            # if output channel name is currently in the roi channels names.
            if new_chan in self.channels:

                if ask == True:

                    answer = input(
                        f"{new_chan} is currently in the roi {self.id}. Replace? (Y/N): "
                    )

                    if answer != "Y" or answer != "y":

                        print(f"{new_chan} wasn't added")
                        # True means not replace
                        return True

                    return False

                else:

                    if display:
                        print(f"{new_chan} is currently in roi {self.id} data channels")

                    return True

        return False

    def load_roi_data(
        self,
        fpaths,
        channels=CHANNELS,
        pfx_sfx=["", ""],
        chans_times={},
        descriptors=None,
        **kwargs,
    ):
        """
        To load just the ROI and not the whole files
        make sure to use prefix and suffix propperly to avoid delete other channels.

        It add just square ROIs. If want to add circular ROIs use flua.add_rois_data()
        after add the square ROIs.

        If you are loading different ROIs for the same data, its more efficient to
        use flua.load_rois_data(rois) instead of this method over each ROI,
        because it reads the images just one time for all the input ROIs.

        fpaths: list of string
            list with the name of the files to be added

        channels: dict
            with the position and name of the channels

        pfx_sfx: list
            prefix and suffix added to each of the given channels to
            store them in the ROI.

        chans_times: dict
            to map channels to a time serie stores in self.times
            e.g. {'chan1': 'times3', 'chan2': 'times1', 'chan4': 'times1'}
            channels not included in the dict will be assigned None.

        descriptors:dict
            extra text to detail the channel content.
            dictionary with structure:  {channel : 'descriptor text'}
        """

        # built the output channels by adding the preffix and suffix to every input channel
        ochans = [pfx_sfx[0] + chani + pfx_sfx[1] for chani in channels.values()]

        # check the new channels are not previously stored in the roi
        if self.check_channels(ochans, ask=True, display=True):
            print("The data wasn't loaded. Please change the output channel names.")
            return ()

        # get the roi coordinate limits
        xlims = self.xlims
        ylims = self.ylims

        data, chans = get_im_data(
            f_names=fpaths,
            channels=channels,
            frame_limits=[ylims, xlims],
            pfx_sfx=pfx_sfx,
            **kwargs,
        )

        # add the values to the roi
        for key in list(data.keys()):
            self.data[key] = data[key]

        # update the channels
        self.update_channels(chans_times=chans_times, descriptors=descriptors)

    def compose_image(self, channels, ofname=None, join_fname=True, **kwargs):
        """
        To compose a RGB or BRG image from input channels.

        channels: dict
            dictionary with the channels names and positions in the RGB output image
            channels keys have to be
            e.g. {0,1,2}

        ofname: string
            filename base to save the image file.
            if indicated, the image will be stored.

        join_fname: bool
            if True, the default filename composition will be join at the end of
            the input ofname.

        **kwargs:
            use this to set the different options and required information of
            compose_image() function.

        """

        if ofname is not None:
            if join_fname == True:
                ofname += f"id{self.id}_"
        image = compose_image(self.data, channels, ofname=ofname, **kwargs)

        return image

    def average_channels_frames(self, frame_ids, channels):
        """
        compute the "averaged frame data" of the indicated frames,
        for each given channel.

        Parameters
        ----------

        data : list or dictionar
            list of channel data names or channels dictionary with its names as keys.

        ids : list or array of integers
            selected frames ids

        channels : list or dictionary
            the channels names/keys to be used.
            in case of dictionarty, the channels positions and names to be used.
            e.g. channels = { 0 : 'R', 1 : 'G'},
            indicates that R and G channels will be used

        **kwargs:
            other options for flua.average_images(), like colormap or
            renormalization parameters.

        Returns
        -------
        avg: dictionary
            averaged image for each indicated channel

        """
        # compute the averge value
        avg_frame = average_images(
            self.data, ids=frame_ids, channels=channels, **kwargs
        )

        return avg_frame

    def subtract(
        self,
        sub_data,
        channels,
        assign=False,
        pfx_sfx=["", "wob"],
        chans_times={},
        descriptors=None,
        **kwargs,
    ):
        """
        to perform subtraction over indicated channels data.

        sub_data are the value(s) to be subtracted, with an array or scalar in
        each dictionary element.

        channels: list o
            with the channels of data to be used

        assign: bool
            if True, the channels will be assigned to the data with the name of
            the original channel plus preffix and suffixs.

        pfx_sfx, chans_times and descriptor same as other functions.

        """
        # built the output channels by adding the preffix and suffix to every input channel
        ochans = [pfx_sfx[0] + chani + pfx_sfx[1] for chani in channels]

        # check the new channels are not previously stored in the roi
        if self.check_channels(ochans, ask=True, display=True):
            print("The data wasn't loaded. Please change the output channel names.")
            return ()

        # convert subtraction data to a dict to be compatible with the used function
        if type(sub_data) != dict:
            dict_data = {}
            for chan in channels:
                dict_data[chan] = sub_data

        sub_data = dict_data

        result_data = subtract_data(self.data, sub_data, channels, **kwargs)

        # if indicated, assign the result as roi.data channels
        if assign:
            # add the values to the roi.data
            for key in list(result_data.keys()):
                self.data[key] = result_data[key]

            # update the channels
            self.update_channels(chans_times=chans_times, descriptors=descriptors)

        else:
            return result_data

    def get_frame(self, channel, frame):
        # for parameters see source get_roi_frame functionn
        return get_roi_frame(self, channel, frame)

    def get_attrs(self, attrs):
        """

        Parameters:
        attrs: list of strings
            list with the names of the attributes of interest

        Return
        ------
        values:
             list with the values of attrs
        """
        values = []
        if type(attrs) != list:
            attrs = [attrs]

        for attr in attrs:
            values.append(getattr(self, attr))
        return values

    def attr_names(self):
        """
        list of object attribute names
        """
        return list(self.__dict__.keys())

    ##################################
    # access other fluopi functions
    def plot_roi_frame(self, **kwargs):

        flup.plot_roi_frame(self, **kwargs)


###########################
####### Functions #########


def round_up(value):
    """
    to round up the given value and convert to integer dtype.
    (integer type is the default in the operating system. np.int_ means that)

    if the input is a np.array or a list, their elements will be int type,
    but rounded is still a np.array or list.

    Parameters:
    ----------
    value: numeric or np.array or list
        any given numeric value

    Return:
    -------
    rounded: integer or np.array (integer type)
        the value rounded to up.
        e.g:

        0.5 is round to 1
        0.4 is round to 0
        -0.5 is round to 0

    value: same as input
        in case is not posible to perform the operation it returns
        the given value.
        e.g. if the given value is None.
    """
    # NumPy array path: one float64 temp, then one int array
    if isinstance(value, np.ndarray):
        tmp = value.astype(np.float64, copy=True)
        np.add(tmp, 0.5, out=tmp)
        np.floor(tmp, out=tmp)
        rounded = tmp.astype(np.int_, copy=False)

    # Python list path: list comprehension, no big arrays
    elif isinstance(value, list):
        rounded = [int(math.floor(v + 0.5)) for v in value]

    # Scalar or other
    else:
        try:
            rounded = int(math.floor(value + 0.5))
        except:
            return value

    return rounded



def missing_intensities(image, step=1, down_range=0, up_range=256):
    """
    To get the missing intensities in the given image array.
    e.g. if the image is uint8 the values range should be [0,256]
    if the image has no pixel with values equal to 25, 71, 36, this
    function outputs that values.


    """
    # define the number of groups
    nbins = round_up((up_range - down_range) / step)  # tipically equal to 256

    # Crear el histograma de valores
    hist, values = np.histogram(image, bins=nbins, range=(down_range, up_range))

    # hist has the counts per value

    # Encontrar los índices donde el histograma es cero
    missing_indexs = np.where(hist == 0)[0]
    missing_values = values[
        missing_indexs
    ]  # obtener los valores asociados a cada indice

    return missing_values


def strip_chars(word, chars):
    """
    to elimnate the each char in the list of chars from the word.

    word: string
        The word to strip
        e.g. '/\/\/name//\/\/'

    chars: list of chars
        e.g. ['/', '\\']

    Return:
        The word without the chars
        e.g. 'name'
    """

    while True:
        # copy
        word2 = (word + ".")[:-1]

        # strip
        for char in chars:

            word = word.strip(char)

        # if doesn't change
        if word == word2:
            break

    return word


def create_path(
    fname, file_ext="", subdirs="", file_mode=False, sep_folder_chars=["/", "\\"]
):
    """
    it create a normalized path for the file name or folder name, based
    on the current working directory, and the given subdirs and file_ext.

    If file_mode = True, it will work on joining the file_ext to the file name.
    if fname includes the extension, it will be used as is, and file_ext will
    be ignored.

    """
    # remove the '/' and '\' at the beggining and at the end of fname and subdirs inputs
    fname = strip_chars(fname, sep_folder_chars)
    subdirs = strip_chars(subdirs, sep_folder_chars)

    if file_mode == True:
        # elimante the dots from file extension borders
        file_ext = strip_chars(file_ext, sep_folder_chars)
        file_ext = file_ext.strip(".")

        # rsplit('.', 1) to split just 1 time from the right limit
        # fname = fname.rsplit('.', 1)[0] # --> to make sure don't use the format in fname

        # if name doesn't come with the format included, use the one in file_ext
        if len(fname.split(".")) == 1:

            fname = fname + "." + file_ext

        elif fname.endswith("."):

            fname = fname + file_ext
        else:
            # Keep the file format indicated in fname
            pass

    # built the absoluthe paths
    fpath = os.path.join(os.getcwd(), subdirs, fname)
    dirpath = os.path.dirname(fpath)

    return (fpath, dirpath)


def create_folder(folder_name, subdirs="", sep_folder_chars=["/", "\\"]):
    """
    it verifies if the folder exists, and if not, it creates it.

    folder_name: string
        name of the folder to create
        e.g. 'folder_name'

    subdirs: string
        subdirectories between the current working directory to new folder
        e.g. subdirs = 'folder1/folder2' for 'folder1/folder2/folder_name'

    sep_folder_chars: list
        list of chars to strip from the folder_name
        e.g. ['/', '\\'] to avoid path separators at borders

    Return
    ------
    folder_path: str
        The absolute and propperly normalized folder path

    """

    # built the absoluthe paths (eliminate folder separatoros from borders too)
    folder_path, dirpath = create_path(
        folder_name, subdirs, sep_folder_chars=sep_folder_chars
    )

    # update the clean folder name
    folder_name = folder_path.split("/")[-1]

    # In case it doesn't exists
    if not os.path.exists(folder_path):

        # create the folder and the required root folder if the case
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' has been created in '{dirpath}'.")

    else:
        # La carpeta ya existe
        print(f"Folder '{folder_name}' already exists in '{dirpath}'.")

    return folder_path


def create_file(
    filename, file_ext="", subdirs="", overwrite=False, sep_folder_chars=["/", "\\"]
):
    """
    it verifies if the filename exists, and if not, it creates the
    required subdirectories where will be stored (in case they doesn't
    exist yet) and returns the complete normalized file_path.

    if the filename exists, it returns False (i.e. doesn't create the file)

    filename: string
        name of the filename to create
        e.g. 'filename.txt'

    subdirs: string
        subdirectories from current working directory to filename
        e.g. subdirs = 'folder1/folder2' for 'folder1/folder2/filename.txt'

    chars: list
        list of chars to strip from the folder_name
        e.g. ['/', '\\'] to avoid path separators at borders

    overwrite: bool
        If False, you will be asked prior to overwrite the file
        If True, the files will be overwrittem directly if there was a previous
        version.
    Returns
    -------
    (file_path, boolean)

    file_path: str
        The absolute and propperly normalized file path

    boolean
        True indicates to create the file (it doesn't exist or overwrite was chosen)
        False to don't create the file
    """
    # built the absoluthe paths (eliminate folder separatoros from borders too)
    file_path, dirpath = create_path(
        filename, file_ext, subdirs, file_mode=True, sep_folder_chars=sep_folder_chars
    )

    # update the clean filename
    filename = file_path.split("/")[-1]

    # In case it doesn't exists
    if not os.path.exists(file_path):

        # In case the target directory doesn´t exist
        if not os.path.exists(dirpath):
            # create it
            os.makedirs(dirpath)

            print(f"'{dirpath}' has been created.")

        print(f"'{filename}' will be created in '{dirpath}'.")

        return (file_path, True)

    else:

        print(f"File '{filename}' already exists in '{dirpath}'.")

        # in case of overwrite, create the files anyways
        ow_message = f"'{filename}' will be overwritten in '{dirpath}'."

        if overwrite:
            print(ow_message)
            return (file_path, True)

        else:
            # ask for replace
            replace = input("Replace it? (Y/N): ")

            # create the file anyways if indicated
            if replace.lower() == "y":

                print(ow_message)
                return (file_path, True)

            # doesn't create the file
            return (file_path, False)


def save_obj(
    obj,
    ofname,
    file_ext="pickle",
    subdirs="",
    sep_folder_chars=["/", "\\"],
    overwrite=False,
):
    """
    To save a pickle file with "obj" object in a desired folder

    if ofname includes the format in its name, that format will be used instead
    of "file_ext" variable.

    Parameters
    ----------
    obj : python object
        python object to be saved as pickle file
        (e.g. dictionay, list, customized objects, etc)

    ofname : string
        output filename with which save the object

    subdirs: string
        subdirectories from current working directory to object filename
        e.g. subdirs = 'folder1/folder2' for 'folder1/folder2/ofname.pickle'

    sep_folder_chars: list
        forbiden strings at the borders
        like the separator character between folder and files

    file_ext: str
        used pickle format extension name
        typically 'pkl' or 'pickle'
        (--> my_file.pickle or my_file.pkl)
        The file_ext extact string doesn't matters but need to be consistent
        with the load.

    overwrite: bool
        If False, you will be asked prior to overwrite the file
        If True, the files will be overwrittem directly if there was a previous
        version.

    Returns
    -------
    None. Just pickle ans save the object.
    """
    # built the absoluthe paths and verify if it previously exists
    fpath, create = create_file(
        ofname,
        file_ext,
        subdirs,
        overwrite=overwrite,
        sep_folder_chars=sep_folder_chars,
    )

    # update the ofname
    ofname = fpath.split("/")[-1]

    # file doesn't exists or overwrite was chosen
    if create == True:

        # save the object
        with open(fpath, "wb") as file:
            pkl.dump(obj, file, pkl.HIGHEST_PROTOCOL)

        print(f"'{ofname}' stored successfully.")

    else:
        # Don't save it
        print(f"'{ofname}' wasn't stored.")


def load_obj(filename, file_ext="pickle", subdirs="", sep_folder_chars=["/", "\\"]):
    """
    To load a pickle object from a desired direction (indicated by subdirs)

    Parameters
    ----------
    filename : string
       filename of the object to be loaded

    subdirs: string
        subdirectories from current working directory to object filename
        e.g. subdirs = 'folder1/folder2' for 'folder1/folder2/filename.pickle'

    file_ext: str
        used pickle format extension name
        typically 'pkl' or 'pickle'
        (--> my_file.pickle or my_file.pkl)
        The file_ext extact string doesn't matters but need to be consistent
        with the load.

    sep_folder_chars: list
        forbiden strings at the borders
        like the separator character between folder and files

    Returns
    -------

    returns the loaded ('unpickled') object

    """

    # compose the absolute path propperly
    fpath, dirpath = create_path(
        filename, file_ext, subdirs, file_mode=True, sep_folder_chars=sep_folder_chars
    )

    # update filename
    filename = fpath.split("/")[-1]

    if os.path.exists(fpath):
        # open the file
        with open(fpath, "rb") as file:

            print(f"'{fpath}' loaded successfully")

            return pkl.load(file)
    else:
        print(f"'{filename}' doesn't exist in '{dirpath}'")
        return False


def save_im(
    image,
    fname,
    fformat="png",
    subdirs="",
    cformat="RGB",
    overwrite=False,
    sep_folder_chars=["/", "\\"],
):
    """
    to save input image (RGB or BRG) as RGB image.
    It also supports monochromatic images.

    image data type has to be 'uin8' [int values from 0 to 255].
    values over 255 will be transformed to 255 and decimal point values are
    round (i.e. images between in space [0,1] are binarized at 0.5 threshold)

    Parameters
    -----------
    image: array like
        2D (i.e. monochromatic) or 3D array in RGB or BRG format.

    fname: string
        name of the file

    fformat: string
        file format extension to save the file

    cformat: string
        options: 'RGB' or 'BRG'
        indicates if the ** input image ** channels order is RBG or BRG.
        for example, RGB means
        R channel is image[:,:,0]
        G channel is image[:,:,1]
        B channel is image[:,:,2]

    """

    # check image format
    dtype = image.dtype

    if dtype != "uint8":

        print(
            f"\n** Warning ** image data type is '{dtype}' "
            "instead of 'uint8'.\nDecimal point values will be round and "
            "values outside [0,255] will by clipped.\n"
        )

    # built the absoluthe paths and verify if it previously exists
    fpath, create = create_file(
        fname, fformat, subdirs, overwrite=overwrite, sep_folder_chars=sep_folder_chars
    )
    # update the fname
    fname = fpath.split("/")[-1]

    # file doesn't exists or overwrite was chosen
    if create == True:

        # change the format if necessary
        if cformat == "RGB" and image.ndim == 3:

            # Convert from RGB to BGR (cv2 requires BRG to save as RGB)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Guardar la imagen con OpenCV
        cv2.imwrite(fpath, image)

        print(f"'{fname}' stored successfully.")

    else:
        # Don't save it
        print(f"'{fname}' wasn't stored.")


def save_figure(
    figure,
    fname,
    file_ext="pdf",
    subdirs="",
    sep_folder_chars=["/", "\\"],
    overwrite=False,
    **kwargs,
):
    """
    To save a matplotlib figure as  a desired folder

    if fname includes the format in its name, that format will be used instead
    of "file_ext" variable.

    Tipycally call it as:
    fig_kwargs = {'bbox_inches': 'tight', 'transparent': True, 'dpi': 300}
    flua.save_figure(fig, ofname, fformat, subfolds,
             overwrite = False, **fig_kwargs)


    Parameters
    ----------
    fig : matplotlib figure
        figure object to be saved as an image file.

    fname : string
        output filename with which save the object

    subdirs: string
        subdirectories from current working directory to object filename
        e.g. subdirs = 'folder1/folder2' for 'folder1/folder2/fname.png'

    sep_folder_chars: list
        forbiden strings at the borders
        like the separator character between folder and files

    file_ext: str
        file format extension to be used. e.g. 'png' for 'image.png'

    overwrite: bool
        If False, you will be asked prior to overwrite the file
        If True, the files will be overwrittem directly if there was a previous
        version.

    kwargs:
        any other parameter accepted by matplotlib.savefig() to customize the
        save process. e.g. transparent=True, bbox_inches='tight', dpi=300, etc.

    Returns
    -------
    None. Just save the figure as an image file.
    """
    # built the absoluthe paths and verify if it previously exists
    fpath, create = create_file(
        fname, file_ext, subdirs, overwrite=overwrite, sep_folder_chars=sep_folder_chars
    )

    # update the fname
    fname = fpath.split("/")[-1]

    # file doesn't exists or overwrite was chosen
    if create == True:

        # save the object
        figure.savefig(fpath, **kwargs)  # transparent=True

        print(f"'{fname}' stored successfully.")

    else:
        # Don't save it
        print(f"'{fname}' wasn't stored.")


def count_files(path, ftype=None, separator=""):
    """
    To count the number of files of a defined extension (filetype on a certain folder (path)

    Parameters
    ----------
    path : string
        folder name where the images are stored

    ftype : string
        extension of the files to count (e.g. tif, png, jpg)
        if None, it counts all the files

    seperator: string
        optional. It is used to split the path and take just the las folder name.

    Returns
    -------
    fcount : int
        number of defined filetype files on the path folder

    """
    # add the dot if ftype doesn´t include it
    if not (ftype.startswith(".")):
        ftype = "." + ftype

    # ImageCount = len(glob.glob1(path,"*."+file_type))
    # get all the files from folder
    files = os.listdir(path)

    # Filtrar archivos por la extensión deseada
    fcount = len([1 for file in files if file.endswith(ftype)])

    print(path.split(separator)[-1] + " = " + str(fcount) + " files")
    return fcount


def extract_number(fpath, npos=-1, init=False, separator="/"):
    """
    To extract the numeric values of a input string.
    It is specially designed to extract numeric values from file paths and
    use the output to sort them.
    For this reason the parameter init, let you manage the fnames that doesn´t
    contain any numeric values on them (and put them at beggining if True,
    and at the end if False).

    Parameters
    ----------
    fpath: string
        actually it can be any string

    npos: integer, -1 or None.
        If the fpath has more than one number, npos indicates the indice
        of the desired numeric value in the extracted list.
         # e.g. numbers = ['42', '20'] if fpath = 'folder/pa42pl20la'.
         if npos = 0 --> 42 will be returned.
        if npos = None, the full list will be returned
        if npos = -1, the last element will be returned.

    init: bool
        Used to manage an fpath that doesn´t has numbers.
        if True, -inf will be returned for this fpath input.

    separator: string
        the string used to split the fpath. The last element
        got after the split is used.

    Return
    ------
    number: number or list
        the extracted number or list of extracted numbers.
        it will depends on the npos and init parameters.

    """

    # extract the filename from filepath if it's the case
    fname = fpath.split(separator)[-1]

    # get the list of numeric characters in the fname string
    numbers = re.findall(r"\d+", fname)

    # in case numbers has elements, return the last element of the list
    if numbers:

        if npos is not None:
            number = int(numbers[npos])
            # e.g. last_number = 20 in the example
        else:
            number = numbers

    # in case it doesn´t
    else:
        # to sort them at the beggining
        if init:
            number = -100000000  # float('-inf')

        # to sort them at the end
        else:
            number = 100000000  # float('inf')

    return number


def print_files(filenames, fpath=None, total_files=None):
    """
    To display the indicated files enumerated

    Parameters
    ----------
    filenames: list
        list of filenames

    fpath: string
        folder name of the given filenames

    total_files: int
        Total number of files in the indicated folder

    """

    nfiles = len(filenames)

    # get the number of digits of the file count
    digits = len(str(nfiles))  # important for the alignement in the display

    if fpath != None:
        print(f"\nPath: {fpath}\n")

    if total_files != None:
        print(
            f"{nfiles} files where selected from the {total_files} files in the folder\n"
        )

    print("\nSelected files: ")

    # print the files
    for idx, file in enumerate(filenames):

        print(f"[{idx:{digits}}] = {file}")


def select_files(
    ftype="",
    fpath=None,
    name_key="",
    start_key="",
    sort=True,
    init=0,
    end=0,
    file_step=1,
    selected_files=None,
    display_files=True,
    separator="/",
    interact=False,
    output=None,
    **kwargs,
):
    """
    Select the files of a defined extension (filetype) and includes
    the input name_key string in its name, from a certain folder (fpath)
    In case you don't specify any key or ftype, you will get all the files
    in fpath.

    Parameters
    ----------
    ftype : string
        desired file extension to be selected.
        e.g '.png'

    fpath: string
        folder path.
        It could be the absolute path (recomended) or a specific sub-folder
        of the current working directory.
        If None, the current working directory will be used.

    name_key: string
        string to filter the elements based on any substring.
        e.g. name_key = 'W1', will select only the filenames
        that contains 'W1' in any part of its name.

    start_key: str
        string to filter the elements based on the initial characters
        of their names.
        e.g. start_key = 'B', will select only the filenames that starts
        with 'B'.

    sort: bool
        if True, the list of selected files will be sorted in ascending way
        based on the number present in its filenames. The files without a number
        will be sort at the end of the list.

    display_files: boolean
        if True the lists of filenames are displayed

    seperator: string
        optional. Folder separator character.

    init: int
        index to start selecting the subset of files

    end:  int
        index to end selecting the subset of files
        end == 0 means until the last file.

    frame_step: int
        step to sepect the subset of files
        e.g. if 2 take one every two files.

    interact: bool
        True to use inside interactive_sfiles().
        False to use directly this function (without use of widgets)

    output: widgets
        In case of widgets usage.
        None by default

    **kwargs:
        to set any other parameters of extract_number() function used for sorting

    Returns
    -------
    sfiles : list
        list of selected filenames
    nfiles: int
        number of selected files

    """

    if fpath is None:
        fpath = os.getcwd()

    # relative path of the given fpath
    rpath = os.path.relpath(fpath) + separator

    ## store and display
    files = os.listdir(fpath)

    sfiles = []
    for file in files:
        if (
            file.endswith(ftype)
            and file.find(name_key) >= 0
            and file.startswith(start_key)
        ):

            # append the file
            sfiles.append(rpath + file)

    ### sort the files based on the number present in its filename ###
    if sort:

        # fname_number = extract_number(fpath, separator = separator, **kwargs)
        sfiles = sorted(
            sfiles, key=lambda x: extract_number(x, separator=separator, **kwargs)
        )

    ### select a subset of the files based on their index
    if end == 0:
        end = len(sfiles)

    ids = np.arange(int(init), int(end), int(file_step))

    sfiles = [sfiles[i] for i in ids if i < len(sfiles)]

    ## display the selected filenames
    if display_files:
        # display
        # in case of interactive selection with widgets
        if interact:
            with output:  # para mantener los widgets
                clear_output(wait=True)  # para eliminar el print anterior
                print_files(sfiles, fpath, len(files))

        # in case of direct use
        else:
            print_files(sfiles, fpath, len(files))

    # Results
    if selected_files == None:

        return (sfiles, len(sfiles))

    else:
        # update the dictionary
        selected_files["fnames"] = sfiles
        selected_files["nfiles"] = len(sfiles)


def interactive_sfiles(
    fpath=None, display_files=True, separator="/", interact=True, **kwargs
):
    """
    Interactive vertion of selected files function
    select_files(
    (ftype = '', fpath = None, name_key='', start_key = '',
                 sort = True, display = True, separator = '/', **kwargs)
    """
    # init the return dictionary
    selected_files = {"fnames": "", "nfiles": 0}

    # Folder path Text Box
    fpath_input = widgets.Text(
        value=fpath,
        placeholder="absolute path",
        description="Folder path:",
        continuous_update=False,
        style={"description_width": "150px"},
        layout=widgets.Layout(width="800px"),
    )

    # Filename extension Text Box
    extension_input = widgets.Text(
        value="png",
        placeholder="e.g. png, jpg, ...",
        description="Filename extension:",
        continuous_update=False,
        style={"description_width": "120px"},
        layout=widgets.Layout(width="400px"),
    )

    # Name Key Text Box
    namek_input = widgets.Text(
        value="",
        placeholder="any key in the name",
        description="Name Key:",
        continuous_update=False,
        style={"description_width": "120px"},
        layout=widgets.Layout(width="400px"),
    )

    # Start Key Text Box
    stark_input = widgets.Text(
        value="",
        placeholder="Any start key in the name",
        description="Start Key:",
        continuous_update=False,
        style={"description_width": "120px"},
        layout=widgets.Layout(width="400px"),
    )

    sort_input = widgets.Checkbox(
        value=True, description="Sort by number in name", disabled=False, indent=False
    )

    init_input = widgets.IntText(
        value=0,
        placeholder="init index of subset",
        description="Init index:",
        continuous_update=False,
        style={"description_width": "150px"},
        layout=widgets.Layout(width="400px"),
    )

    end_input = widgets.IntText(
        value=0,
        placeholder="init index of subset",
        description="End Index:",
        continuous_update=False,
        style={"description_width": "150px"},
        layout=widgets.Layout(width="400px"),
    )

    step_input = widgets.IntText(
        value=1,
        placeholder="step index of subset",
        description="Step Index:",
        continuous_update=False,
        style={"description_width": "150px"},
        layout=widgets.Layout(width="400px"),
    )

    # Vincular la actualización de la lista a los inputs de los widgets
    def on_value_change(change):
        select_files(
            extension_input.value,
            fpath_input.value,
            namek_input.value,
            stark_input.value,
            sort_input.value,
            init_input.value,
            end_input.value,
            step_input.value,
            interact=interact,
            output=output,
            selected_files=selected_files,
            **kwargs,
        )

    # Conectar los eventos de los sliders a la función de actualización
    extension_input.observe(on_value_change, names="value")
    fpath_input.observe(on_value_change, names="value")
    namek_input.observe(on_value_change, names="value")
    stark_input.observe(on_value_change, names="value")
    sort_input.observe(on_value_change, names="value")
    init_input.observe(on_value_change, names="value")
    end_input.observe(on_value_change, names="value")
    step_input.observe(on_value_change, names="value")

    # Cuadro de texto con instrucciones
    instructions = widgets.HTML(
        value="""
        <h3>Instrucciones:</h3>
        <ul>
            <li>Input the parameters to select the desired data.</li>
        </ul>
        """,
        layout=widgets.Layout(width="800px"),
    )

    # Cuadro de texto con instrucciones
    subset_text = widgets.HTML(
        value="""
        <h3>Subset Selection:</h3>
        <ul>
            <li>Enter the proper index of a subset of files [Optional].</li>
        </ul>
        """,
        layout=widgets.Layout(width="800px"),
    )

    # Crear un layout personalizado
    controls = widgets.VBox(
        [
            instructions,
            fpath_input,
            widgets.HBox([extension_input, namek_input, stark_input]),
            sort_input,
            subset_text,
            widgets.HBox([init_input, end_input, step_input]),
        ]
    )

    output = widgets.Output()

    display(widgets.VBox([controls, output]))
    select_files(
        extension_input.value,
        fpath_input.value,
        namek_input.value,
        stark_input.value,
        sort_input.value,
        init_input.value,
        end_input.value,
        step_input.value,
        interact=interact,
        output=output,
        selected_files=selected_files,
        **kwargs,
    )  # lista inicial

    return selected_files


def sublist(files, ids, display=True):
    """
    To select a subgroup of the listed files by their list index.

    Parameters
    ----------

    files: list
        list of filenames
    ids: list or array
        ids of the positions in the list of the desired files to select.
    display: bool
        if True, display the selected files

    Return
    ------
    sfiles: list
        list of selected files
    """

    # select the files based on their index
    sfiles = [files[i] for i in ids if i < len(files)]

    # display if indicated
    if display:

        print_files(sfiles)

    return sfiles


def get_metadata(filename, param_names="", key_sep=":"):
    """
    To get the parameters values from the metadata input filename (.txt).
    The key nomenclature can be customized with the function parameters.
    The default expected estructure of the input file is like this:

    param_1: some_value or values
    param_2: some_value or values
    ...
    param_n: some_value or values

    Parameters
    ----------
    filename: string
        filename to be read

    param_names: string or list of string
        name of the parameter or parameters to return.
        if any parameter is indicated, all are returned.

    key_sep: string
        simbol, character or specific string used to split the parameter name from
        its value.

    Return
    ------

    parameters: dictionary
        Dictionary with the parameters names as keys and their read value

    """
    # correct param_names type if necessary
    if not isinstance(param_names, (list, tuple)):
        param_names = [param_names]

    parameters = dict()

    with open(filename, "r") as f:

        for idx, line in enumerate(f):

            # the parameter name is everything at the left of the first "key_sep"

            pname = line.split(key_sep)[
                0
            ].strip()  # strip eliminate front and back white spaces

            if pname in param_names or param_names == [""]:

                # the value is everything after the first 'key_sep"
                try:
                    # get the value
                    value = line.split(key_sep)[1]
                    value = value.split("\n")[
                        0
                    ]  # eliminate the break line symbol if present
                    value = (
                        value.strip()
                    )  # eliminate front and back white spaces if present
                except:
                    print(
                        f"\n[Error]Cannot be able to split parameter and its value at line {idx}"
                    )
                    print(f"----- Check the separator character in {filename} -----\n")
                    raise

                parameters[pname] = value

    return parameters


def get_times(
    fpaths,
    t0=0,
    param_name="SensorTimestamp",
    mdata_ftype=".txt",
    mdata_fpath=None,
    separator="/",
    convertion=1 / (1000000000 * 3600),
    **kwargs,
):
    """
    to get the time information from metadata files asociated to the
    filenames in the data set. It is assumed the same as the file_names but
    with the extension indicated in mdata_fext
    Use them to construct the propper time vector.

    The expected time information in metadata is like this:
    'SensorTimestamp': '[945443544808000, 945444240081000, 945446033038000]'
     This is the time, ,
    Times, measured in nanoseconds from when the raspberry pi booted,correspond
    to the exact time when the image was taken. Then, you have to specify
    the experimental time value of the first image in the input fpaths
    (typically zero) as the t0 parameter.

    Parameters
    ----------
    fpaths: list of string
        list with the filenames paths to search the time information in its
        asociate metadata file.

    t0: numeric
        time value asociated to the first image in the data set.

    param_name: string
        name of the time parameter to be read from the metadata file.

    mdata_ftype: string
        metadta file extension type

    mdata_fpath: string
        metadata folder path. If None, it is assumed the same as the file_names.

    separator: str
        character used to split the folders

    convertion: numeric
        factor to convert the time values to hours, or any other desired unit.

    kwargs: dictionary
        to pass any other parameter to the function get_metadata

    Return
    ------

    times: list of float
        list with the time values for each asociated input filename, in the
        same order as the input fpaths. The unit values dependes on the
        convertion parameter, but default convert to hour (divides by 10^9
        together 3600)

    """

    # add the dot if the metadata file type doesn´t include it
    if not (mdata_ftype.startswith(".")):
        mdata_ftype = "." + mdata_ftype

    # init the time list
    times = []

    for fpath in fpaths:

        # extract the filename and path from fpaths
        path = fpath.rpartition(separator)[0] + separator
        name = fpath.split(separator)[-1].split(".")[0]

        # use the same path if no other was indicated
        if mdata_fpath is None:
            mdata_fpath = path

        # compose the metadata filename
        mdata_fname = mdata_fpath + name + mdata_ftype

        # get its values from the metadata (it returns a dict)
        dvalues = get_metadata(mdata_fname, param_names=param_name)

        # extract just the strings asociated to numeric values
        values = extract_number(dvalues[param_name], npos=None)

        # convert to number type, just taking the first ndigits
        values = [int(value) for value in values]  # [:ndigits]

        # compute the mean value
        mvalue = np.array(values).mean()

        # append it
        times.append(mvalue)

    times = np.array(times)

    # subtract the epoch time of the minimal value (i.e. the first image) and sum its experiment time
    # and apply the units convertion
    times = convertion * (times - times.min() + t0)

    return times


def im_to_vector(image):
    """
    Convert each image channel to a vector.

    Parameters
    ----------
    image : array like
        image data array. e.g. array of n,m,k dimentions.
        where n,m are image size, and k the number of channels.

    image_count : int
        total number of files on the folder (can be obtained with count_files function)

    f_name : string
        file name pattern including full path where images are stored, e.g. "/folder/image-%04d"

    init: int
        first image number name to be used in the analysis.
        e.g. init = 33 means to use /folder/image-%33

    Returns
    -------
    one of the next outputs depending on the propper case:

    vectors: dictionary
        each channels is stores as a list element in the dictionary

    im_vector: 1d-array
        if the image has just one channel, return directly a vector
        instead of a dictionary

    """

    dims = image.shape
    n_vals = dims[0] * dims[1]

    if len(dims) == 3:
        vectors = {}
        for i in range(dims[2]):

            im_vector = image[:, :, i].reshape((n_vals, 1))
            vectors[i] = im_vector

        return vectors

    else:
        im_vector = image.reshape((n_vals, 1))
        return im_vector


def get_im_data(
    f_names="",
    x_frames=1,
    init=0,
    end=-1,
    channels=CHANNELS,
    get_mono=None,
    pfx_sfx=["", ""],
    frame_limits=[[0, None], [0, None]],
    out_dtype=None,
    renorm=False,
    **kwargs,
):
    """
    Load image data from a sequence of filenames.
    It will be load the data starting from the "init" image number to the
    "end" image number with steps equals to x_frames.
    In case get_mono True, the output channel will keep the same data type
    as the input arrays at least you indicate another one in **kwargs.

    Parameters
    ----------
    f_names : string or list
        file name string pattern including full path where images are stored, e.g. "/folder/image-%04d"
        it could be a list. In that case, it is assumed it is sorted or in the desired order.

    x_frames : int
        step frames (e.g 10 to use only ten to ten images)

    init: int
        first image number name to be used in the analysis.
        e.g. init = 33 means to use /folder/image-%33 as the starting image
        all the previous ones will be omited.

    end : int
        The number of the final image to be considered.
        This image will not necessarilly be loaded. It will deppends on the
        x_frames too. e.g. if end = 15, init = 0 and x_frames = 2,
        the last image loaded will be the 14.

    channels = dict
        dictionary with the channels names and positions to be used.
        e.g. channels = { 0 : 'R', 1 : 'G'}, indicate that only channels
        in positions 0 and 1 of the image will be used, and the name of those
        channels are 'R' and 'G' respectivelly.

    get_mono: bool
        if True the function will return a monocromatic image with the sum
        of the channels indicated in the previous parameter.

    pfx_sfx: list
        Optional.List with two strings, the first is the prefix added to stored
        channels names and the second is the suffix added to them.

    frame_limits: list
        list with two sub lists [[ymin, ymax],[xmin, xmax]], this values
        are the pixel index limits applied to each frame (just that area will
        be loaded)
        by default the whole image is taken: [[0,-1],[0,-1]]

    renorm: boolean
        if True, data is renormalized between the max and min values of the whole.
        It assures to cover the full data range of the given data type.
        Then, if whole data values are between [10 - 500], they are mapped
        to the data type limits (e.g. [0,255] or [0,1]) and each image is mapped
        accord that relation (e.g. image with values between [25,450] will
        have a range between [8,230])

    out_dtype: string or numpy data type
        output data type in case renormalize the data.
        e.g. out_dtype = 'uint8' -> the data will be formated to that type
        if it's not defiened, use keep the input images data tyoe (get it from the first one)

    **kwargs: dict
        aditional keyword arguments for im_to_1chan function, which is
        used to get the monocromatic images.
        (like the "mode" to join the values -sum, mean, etc-, or dtype modification)

    Returns
    -------
    ims_data: dict
        each element is indicated by the channel name and cotains the data
        of that channel for all the images.
        ims_data[chan_name] = [H, W, im_number]

        In case of monocromatic output, chan_name = the joint of the selected
        channels names. e.g. chan_name = 'RG'

    ochans: dict
        with
        dictionary with the output channels names and positions.
    """

    # create a dictionary to store the data
    ims_data = dict()
    ochans = dict()
    selected_fnames = list()

    c_positions = list(channels.keys())

    # get the output channnels prefix and suffix
    pfx, sfx = pfx_sfx[0], pfx_sfx[1]

    # get the frame dimentions limits
    ylims = frame_limits[0]
    xlims = frame_limits[1]

    # round the limits values to be sure they are integers (use the same limits for all images)
    y0 = round_up(ylims[0])
    y1 = round_up(ylims[1])
    x0 = round_up(xlims[0])
    x1 = round_up(xlims[1])

    # classify the **kwargs
    ito1_params = inspect.signature(im_to_1chan).parameters
    ito1_args = {k: kwargs[k] for k in ito1_params if k in kwargs}

    # In case monochromatic image was solicited
    if get_mono:

        # create the name propperly
        mono_name = pfx

        for k in c_positions:

            mono_name += channels[k]

        mono_name += sfx

        # create the output channels dictionary
        ochans = {0: mono_name}

        # init the list
        ims_data[mono_name] = list()

    # Otherwise store each channel as an element of the dictionary
    else:

        ochans = dict()
        for k in c_positions:
            ochan = pfx + channels[k] + sfx

            # output channels names with the same positions as input ones
            ochans[k] = ochan

            # init the list
            ims_data[ochan] = list()

    # define some variables starting values
    number = init
    if end == -1:
        end = len(f_names) - 1

    heigh = 0  # init the variable

    # store the indicated data
    while True:

        # stop if pass the end
        if number > end:
            break

        # get the image number
        if type(f_names) == list:
            fname = f_names[number]

        elif type(f_names) == str:
            fname = f_names % (number)

        else:
            raise Exception(
                "Invalid type of filenames parameter (it has to be str o list)\n"
            )

        # read the image
        im = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

        # if no output data type was indicated, get it from the first image
        if out_dtype == None:

            out_dtype = im.dtype
            print(f"\ninput images data type: {out_dtype}\n")

        # convert None to shape value to avoid "None + 1" Error (for slicing doesnt matters to pass the maximum)
        if y1 == None:
            y1 = im.shape[0]

        if x1 == None:
            x1 = im.shape[1]

        # crop accord input limits
        im = im[y0 : y1 + 1, x0 : x1 + 1, ...]

        # in case of multicromatic image, convert to RGB
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # store the first image heigh (used in the formatting step at the end)
        if number == init:
            heigh = im.shape[0]

        # In case monocromatic image was solicitated
        if get_mono:

            ims_data[mono_name].append(im_to_1chan(im, channels, **ito1_args))
            # --> for sum, uint8 images will probably become uint16

        # Otherwise store each channel independently
        else:

            # add the image data to the dictionary for the selected channels
            for k in c_positions:

                ochan = ochans[k]
                ims_data[ochan].append(im[:, :, k])

        # store the image filename
        selected_fnames.append(fname)

        # update the image number accord the step
        number += x_frames

    ## data formating and data range determination ##

    # init variables to store the min and max values of data
    dmax = 0
    dmin = 0

    assigned = False  # token

    for chan_name in ochans.values():

        # convert each channels from list to numpy array
        ims_data[chan_name] = np.asarray(ims_data[chan_name])

        # check the correct order of the dimmentions
        while ims_data[chan_name].shape[0] != heigh:

            # rotate the dimentions by moving the dim 0 to the end (-1)
            ims_data[chan_name] = np.moveaxis(ims_data[chan_name], 0, -1)

        if renorm:
            ## data range determination ##
            chan_max = ims_data[chan_name].max()
            chan_min = ims_data[chan_name].min()

            # just for the first channel
            if assigned == False:
                dmax = chan_max
                dmin = chan_min

                assigned = True  # change the token

            else:

                # if channels limits are over than stored dmax and dmin values, update them

                if chan_max > dmax:
                    dmax = chan_max

                if chan_min < dmin:
                    dmin = chan_min

    if renorm:

        # if dtype is not a string, obtain its name string
        if type(out_dtype) != str:
            try:
                out_dtype = out_dtype.name
            except:
                out_dtype = np.array([], dtype=out_dtype).dtype.name

        # just in case is an integer data type
        if "int" in out_dtype:

            # get the values range of data type
            type_lims = np.iinfo(out_dtype)

            dtlim_max = type_lims.max
            dtlim_min = type_lims.min

        else:
            # could be this ones or add another parameter with custom values
            dtlim_max = 1
            dtlim_min = 0

        # compute the linear transformation parameters
        m = (dtlim_max - dtlim_min) / (dmax - dmin)
        n = dtlim_min - m * dmin

        for chan_name in ochans.values():
            # grab the full H×W×N array
            arr = ims_data[chan_name]

            # 1) scale & shift everything in-place
            np.multiply(arr, m, out=arr, casting='unsafe')
            np.add(arr, n, out=arr, casting='unsafe')

            # 2) if going from float→int, round in-place
            if np.issubdtype(arr.dtype, np.floating) and "int" in out_dtype:
                np.rint(arr, out=arr)

            # 3) cast once (no extra temp if memory‐layout already ok)
            ims_data[chan_name] = arr.astype(out_dtype, copy=False)

        print(f"Final channel data format: {out_dtype}\n")

    # store the filenames asociated to the selected images
    ims_data["fnames"] = selected_fnames

    return (ims_data, ochans)


def time_vector(data, x_frames, dt):
    """
    ** deprecated function **
    # --> it should be obtained from metadata

    To create the vector of times for the image sequence loaded

    Parameters
    ----------
    data : dictionary
        dictionary with the R G B data of all images

    xframes : int
        step frames used on the analysis (e.g 10 means you are using one every ten to ten images)

    dt : double
        time step of the frames in hour units. It can be obtained from the file used to perform the timelapse.

    Returns
    -------
    T: array_like
        Time vector for the used data (hour units)
    """

    n = len(data["fnames"])  # number of data points
    T = np.zeros((n))
    for i in range(0, n):
        T[i] = (i) * x_frames * dt

    return T


def grid_calibration(
    images, grid_shape, square_size, grid_class="chess", draw_grid=True
):
    """
    get the lens distortion calibration parameters based on calibration
    grid image. This image should be a monochromatic image, otherwise this
    will be transformed to grayscale.

    Chessboard grid images are the recommended for the calibration as they
    are easier and more precissely detected by the algorithm.

    Parameters:
    ----------
    images:
        List with the calibration images.

    grid_shape:
        Tuple representing the shape of the grid. e.g.:
        grid_shape = (7, 7)  for an 8x8 chessboard. In general for chessboard (rows - 1, cols - 1)
        grid_shape = (4, 11) for a 4x11 circle grid

    square_size: numeric
        The physical distance between square corners or circle centers (e.g., in millimeters).
        e.g. square_size = 20

    grid_class: string
        Type of grid pattern. 'chess' or 'dots', according the calibration grid used.
        (if the calibration images contains chessboard or dots grids)

    draw_grid: boolean
        True to draw the identified grid over the input images.

    Returns:
    --------
    ret:
        Calibration success flag.

    cam_matrix:
        Camera matrix.

    dist_coeffs:
        Distortion coefficients.

    rvecs:
        Rotation vectors.

    tvecs:
        Translation vectors.
    """

    # Arrays to store object points and image points from all the images
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    # Define the object points for the grid pattern in 3D
    objp = np.zeros((grid_shape[0] * grid_shape[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : grid_shape[1], 0 : grid_shape[0]].T.reshape(-1, 2) * square_size
    )

    for i, image in enumerate(images):

        # in case the image is non-monochromatic
        if image.ndim >= 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if grid_class == "chess":
            found, corners = cv2.findChessboardCorners(image, grid_shape, None)
            # Refining corner locations for higher accuracy
            if found:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria=(
                        cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER,
                        30,
                        0.001,
                    ),
                )
        elif grid_class == "dots":

            found, corners = cv2.findCirclesGrid(
                image, grid_shape, flags=cv2.CALIB_CB_SYMMETRIC_GRID
            )

        else:
            raise ValueError('Enter a valid "grid class" parameter option')

        if found:
            img_points.append(corners)
            obj_points.append(objp)

            # Draw and display the pattern on the images for verification
            if draw_grid:

                # cv2.drawChessboardCorners(image, grid_shape, corners, found)
                # cv2.imshow('Grid Detection', image)
                # cv2.waitKey(500)

                # convet to RGB again
                # im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Plotting the detected pattern with matplotlib
                plt.figure(figsize=(8, 8))
                plt.imshow(image)  # or im_rgb
                plt.scatter(
                    corners[:, 0, 0],
                    corners[:, 0, 1],
                    c="r",
                    s=10,
                    label="Detected Corners",
                )
                plt.title(f"Pattern Detection in image {i}")
                plt.legend()
                plt.axis("off")
                plt.show()

        else:
            print(f"Pattern not found in image {i}")

    if found:
        # perform the calibration
        ret, cam_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image.shape[::-1], None, None
        )

        return (ret, cam_matrix, dist_coeffs, rvecs, tvecs)

    else:
        print("No pattern found, it is not possible to perform the calibration")
        return (None, None, None, None, None)


def mean_value(
    p1,
    p2,
    data,
    im_number,
    channels=CHANNELS,
    colors=COLORS,
    read_im=False,
    ylims=None,
    show_std=False,
    show_fig=True,
    data2=None,
):
    """
    compute the background mean value for each channel and frame based on a rectagle
    defined by the user. Plot the rectangle over an image and makes plots of each channel
    mean background value over time.
    By default it uses the values from the indicated channels of data to compose
    the image and get the background values.

    Parameters
    ----------
    p1,p2: list of int values
        rectangle area limits:
        p1 = [y1,x1] = left-up corner.
        p2 = [y2,x2] = right-bottom corner
        Take in account that images are inverted in y-axis, with upper left
        corner as the (0,0) --> then y2 > y1

    data : dictionary
        images data to get the background, and his names on data['fnames']
        each image channel is an elements of the dictionary.

    im_number : int
        image number used to display the rectangle
        if used the total number of files on the folder (which be obtained with
        count_files function), subtract (-1) to its value.

    channels: dict
        dictionary with the channels names and positions to be used.
        e.g. channels = { 0 : 'R', 1 : 'G'}, indicate that only channels
        in positions 0 and 1 of the image will be used, and the name of those
        channels are 'R' and 'G' respectivelly.

    colors: dict
        colors to be used for the respective channel position in the plots.
        e.g. {0:'r', 2: 'b'}, will display a red line for channel 0 and a
        blue line for channel 2.

    read_im: bool
        if True, read the image and plot it.
        if False, plot the values from indicated channels of data.

    show_std : bool
        if True, the std deviation is computed and displayed

    show_fig: bool
        if True, the figure is displayed, otherwise just get return the
        computed values.

    data2: dictionary, optional
        If given, this data is used to plot a second group of mean and statistics,
        along with the first data. This is not displayed as image, just in the
        plot.
        The structure is the same as data.

    Returns
    -------
    data_mean: dictionary
        mean value of each channel for every time frame

    data_std: dictionary
        standard deviation value of each channel for every time frame

    """
    # eg: p1 = [500, 1500] , p2 = [1200, 800]

    y1, y2 = p1[0], p2[0]
    x1, x2 = p1[1], p2[1]

    h = y2 - y1
    w = x2 - x1

    print(h, w)

    if show_fig:
        # plot the defined area
        fig, ax = plt.subplots(2, 1, figsize=(8, 10), layout="constrained")

    # get the channels positions
    c_positions = list(channels.keys())

    # In case it is desired to use the image directly from the folder
    if read_im:

        # get image filename
        try:
            fname = data["fnames"] % (im_number)
        except:
            fname = data["fnames"][im_number]

        # read the image
        im = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

        # in case of multicromatic image, convert to RGB
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # In case to use the information from the channels of data
    else:
        nchans = len(c_positions)

        if nchans == 1:
            im = data[channels[c_positions[0]]][:, :, im_number]
        else:
            im = np.zeros(
                (
                    data[channels[c_positions[0]]].shape[0],
                    data[channels[c_positions[0]]].shape[1],
                    nchans,
                )
            )
            for i in range(nchans):
                chani = channels[c_positions[i]]
                im[:, :, i] = data[chani][:, :, im_number]
    if show_fig:
        # plot the image
        ax[0].imshow(im)
        # rectangle is defined from left-bottom to right-up corner
        rect = Rectangle((x1, y2), w, -h, linewidth=1, edgecolor="r", facecolor="none")
        ax[0].add_patch(rect)

        # get the ylims if were not indicated
        if ylims is None:
            ylims = [int(np.floor(im.min() + 0.5)), int(np.floor(im.max() + 0.5))]

    # get the mean background value at each time for each channel and plot it
    data_mean = {}
    data_std = {}  # also its standard deviation

    for pos in c_positions:

        # get channel selected area values
        chan = channels[pos]
        sub_area_values = data[chan][
            y1:y2, x1:x2, ...
        ]  # .reshape(-1, data[chan].shape[-1])

        # compute the mean and std
        data_mean[chan] = sub_area_values.mean(axis=(0, 1))  # (axis=0)
        data_std[chan] = sub_area_values.std(axis=(0, 1))  # (axis=0)

        if show_fig:
            # mean curve
            ax[1].plot(data_mean[chan][:], colors[pos], label=f"Mean {chan}")

            # std area
            if show_std:
                ax[1].fill_between(
                    range(len(data_mean[chan])),
                    data_mean[chan] - data_std[chan],
                    data_mean[chan] + data_std[chan],
                    color=colors[pos],
                    alpha=0.2,
                    label=f"Std {chan}",
                )

    # in case a second data was given
    if data2 is not None:
        d2_mean, d2_std = mean_value(
            p1,
            p2,
            data2,
            im_number,
            channels,
            colors,
            read_im=False,
            show_fig=False,
            data2=None,
        )

        for pos in c_positions:

            # plot channel selected area values
            chan = channels[pos]

            light_color = flup.lighten_color(colors[pos])

            ax[1].plot(d2_mean[chan][:], color=light_color, label=f"Mean d2 {chan}")
            ax[1].fill_between(
                range(len(d2_mean[chan])),
                d2_mean[chan] - d2_std[chan],
                d2_mean[chan] + d2_std[chan],
                color=light_color,
                alpha=0.2,
                label=f"Std d2 {chan}",
            )
    # display the figure
    if show_fig:
        ax[1].set_title("Selected area mean value over time")
        ax[1].set_xlabel("Frame")
        ax[1].set_ylabel("Mean pixel value")
        if ylims is not None:
            ax[1].set_ylim(ylims)
        ax[1].legend()
        plt.show()

    return (data_mean, data_std)


def average_images(
    data, ids, channels=CHANNELS, factor=None, dtype=None, cmap="viridis", display=False
):
    """
    average the input images data and return the averaged image

    Parameters
    ----------

    data : dictionary
        channels data of images and his names on data['fnames']

    ids : list or array of integers
        selected images ids

    channels : dictionary
        dictionary with the channels positions and names to be used.
        e.g. channels = { 0 : 'R', 1 : 'G'},
        indicates that R and G channels will be used

    factor: int
        factor to reconvert the images values

    cmap: string
        matplotlib colormap name to display the images

    Returns
    -------
    avg: dictionary
        averaged image for each indicated channel

    """
    # convert channels to list if it is the case
    if type(channels) == dict:
        channels = list(channels.values())

    # init the average dictionary
    avg = {}

    # get the number of channels
    nchans = len(channels)

    if display:
        # init the figure
        fig, axs = plt.subplots(
            1, nchans, figsize=(4 * nchans, 3), layout="constrained"
        )

        if nchans == 1:
            axs = [axs]

        for i in range(nchans):
            chani = channels[i]

            # keep the inputa data dtype if no dtype was indicated
            if dtype is None:
                dtype = data[chani].dtype.name

            # init the array to ensure it is float64 and avoid data lost by overflow
            avg[chani] = np.zeros((data[chani].shape[0], data[chani].shape[1]))

            for num in ids:
                avg[chani] += data[chani][:, :, num]  # do this with np.dot

            avg[chani] = avg[chani] / len(ids)

            ## convert the dtype and renormalize by factor if necesary (if factor = 1, it just check/change the dtype)

            # define factor value for the data type check/return/change
            if factor is None:

                if np.issubdtype(dtype, np.integer) and avg[chani].max() <= 1:
                    factor = 255
                else:
                    factor = 1

            avg[chani] = renormalize(avg[chani], factor=factor, dtype=dtype)

            # plot the image channel
            imi = axs[i].imshow(avg[chani], cmap=cmap)
            axs[i].set_title(f"{chani} channel")

            fig.colorbar(imi, fraction=0.035)

        fig.suptitle(" Averaged ids images ", fontsize=12)
        plt.show()

    else:

        for i in range(nchans):
            chani = channels[i]

            # keep the inputa data dtype if no dtype was indicated
            if dtype is None:
                dtype = data[chani].dtype.name

            # init the array to ensure it is float64 and avoid data lost by overflow
            avg[chani] = np.zeros((data[chani].shape[0], data[chani].shape[1]))

            for num in ids:
                avg[chani] += data[chani][:, :, num]  # do this with np.dot

            avg[chani] = avg[chani] / len(ids)

            ## convert the dtype and renormalize by factor if necesary (if factor = 1, it just check/change the dtype)

            # define factor value for the data type check/return/change
            if factor is None:

                if np.issubdtype(dtype, np.integer) and avg[chani].max() <= 1:
                    factor = 255
                else:
                    factor = 1

            avg[chani] = renormalize(avg[chani], factor=factor, dtype=dtype)

    return avg


def subtract_data(
    data,
    bg_data,
    channels=CHANNELS,
    dtype=None,
    factor=1,
    bg_vector=None,
    operation="+",
):
    """
    ## mejorar y verificar esta función para que se adapte bien a los diferentes casos
    ### tanto de data como de bg_data. en este momento bg_data no puede ser un vector!

    Subtract given background value to the data.
    It could be a single value, a background frame to substract to all
    the others o an array of the same size as data (a specific background frame
    for each data frame)
    data and bg_data has to have the same channels keys.
    if "dtype" is indicated it is used along with "factor"to change the data
    type of each channel if it another dtype after the subtraction.

    Parameters
    ----------


    bg_data: dictionary
        estructure [channel][scalar], [channel][h,w] or [channel][h,w,frames]
        Not sure if the case [channel][frames] is working.

    bg_vector: 1D np.array
        vector with a scalar value for each frame.
        These values are used to ponderate the bg_data value for each frame.
        This only works in the case bg_data[channel][h,w].

    operation: string
        '+' to perform addition of the bg_vector values
        '*' to perform multiplication of the bg_vector values

    Returns
    -------
    wob: dict
        dictionary with the indicated channels of data subtracted by
        the given brackground values.

    """
    if type(channels) == dict:
        channels = list(channels.values())

    # init the dictionary
    wob = {}

    for chani in channels:

        bgchani = bg_data[chani]

        chan_dtype = data[chani].dtype  # get the data type

        # check the channel background values data type
        if type(bgchani) != np.ndarray:
            bgchani = np.asarray(bgchani)

            if bgchani.ndim == 0:
                bgchani = np.asarray(bgchani)[np.newaxis]

        # in case background is not a scalar but has 1 dimention less than data
        if data[chani].ndim > bgchani.ndim and bgchani.shape[0] > 1:

            if bg_vector == None:

                print("Using a same background for all frames")
                bgchani = bgchani[:, :, np.newaxis]

        #            else:
        #                try:
        #
        #                    print('Using a specific background for each frame')
        #
        #                    bgchani = bgchani[:, :, None] * bg_vector[chani]
        #
        #                    # correct the data type (and reduce memory usage)
        #                    bgchani = change_dtype(bgchani, chan_dtype)
        #
        #
        #                except:
        #                    raise Exception("Something is wrong with the bg_vector.")

        # make a especial case for background channel frame personalizad for each frame based on a bg_vector
        if data[chani].ndim == 3 and bgchani.ndim == 2 and bg_vector != None:

            print("Using a specific background for each frame")

            # init the output array
            wob[chani] = np.ones_like(data[chani])
            chan_dtype = data[chani].dtype

            # To make the proccess less RAM intensive, we are going to use a for loop
            for i in range(data[chani].shape[2]):

                # compute the frame background and make sure its data type is appropiate
                if operation == "*":
                    bgi = bgchani * bg_vector[chani][i]

                elif operation == "+":
                    bgi = bgchani + bg_vector[chani][i]

                bgi = change_dtype(bgi, chan_dtype)

                # store a mask with True for positions bigger than background and False if lower
                mask = data[chani][:, :, i] > bgi

                # perform the subtraction
                wob[chani][:, :, i] = data[chani][:, :, i] - bgi

                # make value = 0 the positions False in the mask
                wob[chani][:, :, i] = np.where(mask, wob[chani][:, :, i], 0)
        else:
            # store a mask with True for positions bigger than background and False if lower
            mask = data[chani] > bgchani

            # perform the subtraction
            wob[chani] = data[chani] - bgchani

            # make value = 0 the positions False in the mask
            wob[chani] = np.where(mask, wob[chani], 0)

        # convert negtive values to zero
        # wob[chani]  = np.clip(wob[chani] , a_min=0, a_max=None)

        # change the data type if necessary
        if dtype is not None:
            if wob[chani].dtype != dtype:
                wob[chani] = renormalize(wob[chani], factor=factor, dtype=dtype)

    return wob


def bg_subst(data, bg, channels=CHANNELS):
    """
    Substract the mean background value for each channel and frame obtained with BG_Val function.
    (a escalar valur for each channel an frame)

    Parameters
    ----------
    data: dictionary
        containing the channels images data in each element
        data['Channel_name'] = channels_images_array (h,w,n), where n is the number of images.

    bg : array
        background mean value of each channel for every time frame (can be obtained with BG_Val function)

    channels = dict
        dictionary with the channels names and positions to be used.
        e.g. channels = { 0 : 'R', 1 : 'G'}, indicate that only channels
        in positions 0 and 1 of the image will be used, and the name of those
        channels are 'R' and 'G' respectivelly.


    Returns
    -------
    Data: dictionary
        R G B images data with the background substracted

    """

    L = bg[channels[0]].shape[0]
    S1, S2, _ = data[channels[0]].shape

    for key in channels.keys():
        c = channels[key]
        for i in range(0, L):
            BGM = np.ones((S1, S2))
            BGM = BGM * bg[c][i]  # create a matrix with bg to substract it to the frame

            Data = data[c][:, :, i]

            Data = Data - BGM  # perform the substraction

            Data[Data < 0] = 0  # values < 0 are not allowed --> transform it to 0

            data[c][:, :, i] = Data  # actualize Data

    return data


def subtract_to_frame(data, value):
    """
    Subtracts a value or array of values from a frame data.
    It is case sensitive on the dimentions nature of subtracted value

     Parameters:
     -----------
     data: ndarray
        ROI channel data ndarray of dimensions NxMxL.
     value: It can be a scalar, a vector of length L, an NxM matrix, or an NxMxL array.

     Returns:
     --------
     Resulting arrangement after performing subtraction.
    """
    dtype = data.dtype
    # result = np.zeros(data.shape, dtype = 'float32')

    # modify the value according it dimentions
    if np.isscalar(value):
        # subtract the scalar to the whole data
        pass

    elif value.ndim == 1:
        # subtract correponding vector scalar value to each slice in data L dimention
        value = value[np.newaxis, np.newaxis, :]

    elif value.ndim == 2:
        # subtract the matrix  value to each slice in data L dimention
        value = value[:, :, np.newaxis]

    elif value.ndim == 3 and value.shape == data.shape:
        # subtract the correponding matrix to each slice in data L dimention
        pass

    else:
        raise ValueError("Value dimensions are not supported by data")

    # store a mask with True for positions bigger than background and False if lower
    mask = data > value

    # perform the subtraction
    result = data - value
    # make value = 0 the positions False in the mask
    result = np.where(mask, result, 0)

    return result


def data_sum_time(data, channels=CHANNELS):
    """
    Sum the data for each pixel over time

    Parameters
    ----------
    Data: dictionary
        R G B images data

    channels:  dict
        dictionary with the channels names and positions to be used.
        e.g. channels = { 0 : 'R', 1 : 'G'}, indicate that only channels
        in positions 0 and 1 of the image will be used, and the name of those
        channels are 'R' and 'G' respectivelly.

    Returns
    -------
    SData: array like
        Sum data over time and over channels for each pixel of the Data

    """
    c_positions = list(channels.keys())
    added_chans = ""
    for k in c_positions:
        chan_name = channels[k]
        added_chans += str(chan_name)

        # for the rest of the channels
        try:
            SData = SData + data[chan_name][:, :, :].sum(axis=(2))

        # for the first channel
        except:
            SData = data[chan_name][:, :, :].sum(axis=(2))

    plt.imshow(SData)
    plt.colorbar()
    plt.title("Channels " + added_chans)

    return SData


def renormalize(array, vlims=None, factor=None, dtype=None, print_type=False):
    """
    to renomarlize any numeric array.
    Be carefull with the array (or image) formats and normalization factor.
    by default it keeps the data type of the input array.
    In case vlims is not specified, it is set to [0,255] for integers dtype,
    and to [0,1] for floating dtypes.

    if factor is indicated, the data is renormalized by mutltiplying it with
    the factor and reconverting to the propper dtype.
    ** factor is used just in case vlims is not indicated **

    Parameters
    ----------
    array: array like
        multichannel or monocromatic image (or image channel).
        The renormalization is based on the whole arrays values, and not
        per any especific dimention.
        If want to normalize each channels separatelly, just run
        the function for each of them independently.

    vlims: list
        it cotains the renormalization limits values [vmin, vmax]
        it assures the output cover the full range at difference of using factor

    factor: numeric
        value used to ponderate the renormalized image
        e.g. factor = 255 to set this value as the maximum.
        By default it is set as 255 for uint8 data and as 1 for the rest

    dtype: data type indicator (str or dtype)
        if given, the image is reconverted to that format.
        e.g. dtype = np.uint8  will convert the data to 8 bits integer.
        by default it preserves the input image format.

    print_type: bool
        if True, the finally used data type will be print (after pass
        through the change_dtype function).

    Returns
    -------
    narr: array like
        renormalized array to factor*[0,1]
        the format will be np.float64 at least you specify something else
        If you are working in the [0-255] space is better to use
        dtype = np.int8 to reduce memory usage.

    """

    # keep the data type in case it wasn´t indicated.
    if dtype is None:
        dtype = array.dtype.name

    # if dtype is not a string, obtain its name string
    if type(dtype) != str:
        try:
            dtype = dtype.name
        except:
            dtype = np.array([], dtype=dtype).dtype.name

    # in case no value was indicated for factor and vlims
    # or in case a value was indicated for both ---> just use vlims!
    if factor is None or (factor is not None and vlims is not None):

        if vlims is None:
            # define vlims for integer data types
            if np.issubdtype(dtype, np.integer):

                vlims = [0, 255]

            # othewise
            else:
                vlims = [0, 1]

        # perform the normalization to [0,1]
        span = array.max() - array.min()

        # avoid divition error
        if span == 0:
            span = 1

        narr = (
            array - array.min()
        ) / span  # the divition makes it float, avoiding overflow of some formats in next step

        # then, renormalize to vlims
        span = vlims[1] - vlims[0]
        narr = span * narr + vlims[0]

        # ensure the data type is the appropiate and don't generate data lost.
        # Also for integer data type (uint8, int32, etc) round the values instead of floor them
        narr = change_dtype(narr, dtype, print_type=print_type)

        # inform the change if it's the case
        if narr.dtype != dtype:
            print(
                f"renormalized array.dtype is '{narr.dtype}' instead of '{dtype}' to avoid data lost"
            )

        return narr

    # in case vlims was not indicated, check for factor normalization
    elif vlims is None:

        # if a numeric value was indicated for factor
        if isinstance(factor, (int, float, complex, np.number)) and not isinstance(
            factor, bool
        ):
            narr = array / 1  # convert to float
            narr = factor * narr

            # ensure the data type is the appropiate and don't generate data lost.
            # Also for integer data type (uint8, int32, etc) round the values instead of floor them
            narr = change_dtype(narr, dtype, print_type=print_type)

            # inform the change if it's the case
            if narr.dtype != dtype:
                print(
                    f"renormalized array.dtype is '{narr.dtype}' instead of '{dtype}' to avoid data lost"
                )

            return narr


def compose_image(
    data,
    channels,
    data_frame=-1,
    im_structure="RGB",
    dtype="uint8",
    monochromatic=False,
    normalize=255,
    ofname=None,
    fformat=".png",
    overwrite=False,
    **kwargs,
):
    """
    to compose a RGB image from given data and channels.
    if a channel position is not indicated or cannot be added, will be filled
    with zeros.

    Parameters
    ----------

    data: dict
         is a dictionary with the data to compose the image.
         estructure: {channel_name : data_array},
         with data_array shape [N,M] or [N,M,Frames]

    channels: dict
        dictionary with the channels positions 0,1,2 asociated to the names
        of the channels in data to built the image.
        e.g. channels = { 0 : 'CX', 1 : 'nG', 2 : 'C2'}, indicate the image
        will be composed with the channels CX, G and C2 of the data in positions
        0, 1 and 2 of the new image.

    data_frame: integer
        is not necessary for channels with just 1 frame

    im_structure: string
        'RGB' or 'RGBA'

    monochromatic : boolean or str
        This is considered just in case one data channel was required.
        By default is False and no colormap is applied, filling the empty channels with zeros.
        if True the image is generated as monchromatic (gray)
        If you desire another colormap input the name (e.g. 'viridis').
        used in array_to_im() function

    normalize : bool or numeric
        used in array_to_im() function
        if True, each image is normalized to cover full range of colormap.
        If normalize is numeric, this values is used to normalize the image
        between [0, array.max()/normalize]
        In case array.max()/normalize is bigger than 1, the data is renormalized
        to be between [0,1] range.

    **kwargs:
        see save_im function

    Returns
    -------
    imdata: np.array (N,M,3 or 4)
        the output image data array

    """
    # get the channels positions
    c_positions = list(channels.keys())

    # get actual data channels
    dchans = list(data.keys())

    # check if input channels are in data channels
    checked_pos = []
    for pos in c_positions:

        chan = channels[pos]

        if chan not in dchans:
            print(
                f"Channel {chan} cannot be found in dataset. It will be added as zeros."
            )
        else:
            checked_pos.append(pos)

    # if dtype is not a string, obtain its name string
    if type(dtype) != str:
        try:
            dtype = dtype.name
        except:
            dtype = np.array([], dtype=dtype).dtype.name

    # create the array to rebuilt the image based on the first checked channel
    n, m = data[channels[checked_pos[0]]].shape[
        0:2
    ]  # take just the first two dimentions

    if len(c_positions) > 1 or monochromatic is False:
        # in this case, empty channels are left as zeros

        if im_structure == "RGBA":
            imdata = np.zeros((n, m, 4), dtype=dtype)

        else:
            imdata = np.zeros((n, m, 3), dtype=dtype)

        # compose an np.array of the image based on the verified channels
        for i in checked_pos:

            chan = channels[i]

            data_chani = data[chan]

            if data_chani.ndim == 3:

                # get the number position of "-1" frame for display and filename purposes
                if data_frame == -1:

                    data_frame = data_chani.shape[2] - 1

                try:
                    imdata[:, :, i] = data_chani[:, :, data_frame]
                except:
                    print(f"frame {data_frame} cannot be found for channel '{chan}'")

            elif data_chani.ndim == 2:
                imdata[:, :, i] = data[chan][:, :]

    else:  # monochromatic is True

        # in this case the image is converted to multichannel using the input colormap
        imdata = np.zeros(
            (n, m), dtype=dtype
        )  # just to be clear about the dimentions in the code

        chan = channels[checked_pos[0]]
        data_chan = data[chan]

        if data_chan.ndim == 3:

            # get the number position of "-1" frame for display and filename purposes
            if data_frame == -1:

                data_frame = data_chan.shape[2] - 1

            try:
                imdata[:, :] = data_chan[:, :, data_frame]
            except:
                print(f"frame {data_frame} cannot be found for channel '{chan}'")

        elif data_chan.ndim == 2:
            imdata[:, :] = data[chan][:, :]

        # in case a specific colormap was indicated
        if type(monochromatic) is str:

            # conver to RGB/RGBA usint the indicated colormap
            imdata = array_to_im(
                imdata,
                cmap_name=monochromatic,
                normalize=normalize,
                imformat=im_structure,
            )

    # save the image file if indicated
    if ofname is not None:
        fname = f"{ofname}_fr{data_frame:03d}"
        save_im(imdata, fname, fformat=fformat, overwrite=overwrite, **kwargs)

    return imdata


def smooth_data(
    data,
    ksize,
    sigma=0,
    channels=CHANNELS,
    display=True,
    cmap="viridis",
    renorm=False,
    do_uint8=False,
    factor=None,
):
    """
    Apply gaussian filter to smooth each frame data and renormalize
    each channel to cover full data type span if indicated.

    Parameters
    ----------
    data: dictionary
        dictionary with the images data frames stored stored independently for
        each channel.
        e.g. data['chan_name'] = [h,w,n]
        where h and w are image dimentions (heih and wide) and n is the number of frames.
        It has to be 'uint8' format. If not, will be converted prior the
        smoothing.

    ksize: int
        it must be an odd integer number.
        This is the size of the gaussian kernell.
        values between 5 and 15 are typicall. The smoothing increase with this value.

    sigma: double
        Filter parameter (standard deviation).
        If sigma = 0, it is automatically determined according the ksize value.

    channels: dict
        dictionary with the channels positions and names.
        e.g. channels = { 0 : 'R', 1 : 'G'}, indicate the image has the channels
        R and G inpositions 0 and 1 respectivelly.

    display: bool
        if True, the smmoothing effect over the last image is displayed.

    cmap: colormap
        colormap to be used in the image display and colormaps.

    renorm: boolean
        if True, renormalize each channel's data to cover the full data type span
        (full span just refers to [0,1] or [0,255] depending on the data nature)
        It is performed for each channel indenpendently.

        e.g. if data of "channel_i" is between [0.1,0.8] it will make it cover
            the [0,1] full range.
        e.g. if data of "channel_i" is between [10,180], or [2,900], it will
            make it cover the [0,255] full range

    do_uint8: boolean
        if True, convert the data to uint8 (or make sure the data is uint8).
        It basically multiplies the data by 'factor' and then changes its data type
        If 'factor' is not given, it uses 255 if data span [0,1] or just 1 if
        data is in the 255 space.

    factor: numeric
        Optional. Factor used to convert the data to uint8 space (i.e. 0 to 255)
        in case it is in another data type format. By default it uses 255 if the
        max data < 1, and factor = 1 if max data is over 1.

    Returns
    -------

    sdata: dictionary
        Smoothed data per channel per frame (call it as simsT[channel][row,column,frame])

    """
    c_positions = list(channels.keys())

    # init the output dictionary
    sdata = {}

    h, w, n = data[channels[0]].shape

    for chan in channels.values():

        dchan = data[chan]
        data_type = dchan.dtype.name

        if do_uint8:

            # make sure the channel is uint8
            if dchan.dtype != "uint8":

                # in case reconvertion factor wasn´t indicated
                if factor is None:
                    if dchan.max() < 1:

                        # change the space to 0-255 (without renormalization)
                        factor = 255

                    else:
                        # just change the data type
                        factor = 1  # factor = 1 doesn´t have any other effect

                # perform the factor renormalization and/or data convertion
                # this renormalization instance doesn't make the data to cover the full span of values
                dchan = renormalize(dchan, factor=factor, dtype="uint8")

                # update the data type
                data_type = dchan.dtype.name

        # init the channel array
        sdata[chan] = np.zeros((h, w, n), dtype=data_type)

        # go for each frame
        for i in range(n):

            frame = dchan[:, :, i]
            # apply the gaussian filter
            sim = gaussian_smooth(frame, ksize, sigma)
            # sim = cv2.GaussianBlur(frame, (ksize, ksize), sigma)

            # if renormaliztion was indicated, make the data cover the full data type span.
            if renorm:
                sim = renormalize(sim)

            sdata[chan][:, :, i] = sim

    if display:

        nchans = len(c_positions)

        # start the figure
        fig, axs = plt.subplots(
            1, nchans, figsize=(4 * (nchans), 4), layout="constrained"
        )
        fig.suptitle("Smoothed with ksize = " + str(ksize), fontsize=12)

        # plot each channel of the last image
        for pos in c_positions:
            chan = channels[pos]
            last_frame = sdata[chan][:, :, -1]

            if nchans > 1:

                imi = axs[i].imshow(last_frame, cmap=cmap)
                axs[i].set_title(f"Channel {chan}")
                fig.colorbar(imi, fraction=0.035)

            else:
                imi = axs.imshow(sdata[chan][:, :, -1], cmap=cmap)
                axs.set_title(f"Channel {chan}")
                fig.colorbar(imi, fraction=0.035)

    return sdata


def gaussian_smooth(image, ksize, sigma=0):
    """
    Apply a gaussian smoothing filter to the input image.
    This function as the adventage of round the values externally to the
    GaussianBlur function, as its internal rounding method generates some
    loss of values.
    With this function the rounding of values is applyed externally by the
    round_up function.

    Parameters
    ----------
    image: array like

    ksize: int
        it must be an odd integer number.
        This is the size of the gaussian kernell.
        values between 5 and 15 are typicall. The smoothing increase with this value.

    sigma: double
        Filter parameter (standard deviation).
        If sigma = 0, it is automatically determined according the ksize value.

    Returns
    -------
    sim: array
        smoothed image

    """
    # get the input image data type
    im_dtype = image.dtype

    # in case is not float convert to float32
    if im_dtype != "float32" and im_dtype != "float64":

        image = image.astype(np.float32)

    # smooth the image
    sim = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # return to input image data type in case it was changed
    # this just happens if the input type was integer
    if sim.dtype != im_dtype:
        # round the values
        sim = round_up(sim)

        # get the input data type limits
        info = np.iinfo(im_dtype)

        # clip the values to make sure are in range and then convert to original data type
        sim = np.clip(sim, info.min, info.max).astype(im_dtype)

    # return the smoothed image
    return sim


def smooth_im(
    image,
    ksize,
    sigma=0,
    mono_mode="sum",
    channels=CHANNELS,
    display=True,
    cmap="viridis",
):
    """
    Apply a gaussian smoothing filter to the input image and display it if indicated.
    if multichromatic, it also peform the smoothing over the sum of the channels
    and return this data together its renormalized version.

    Parameters
    ----------
    image: array like

    ksize: int
        it must be an odd integer number.
        This is the size of the gaussian kernell.
        values between 5 and 15 are typicall. The smoothing increase with this value.

    sigma: double
        Filter parameter (standard deviation).
        If sigma = 0, it is automatically determined according the ksize value.

    mono_mode: string
        mode to convert the image to monochromatic.
        options: 'sum', 'mean', 'renormalize'

    channels: dict
        dictionary with the channels positions and names.
        e.g. channels = { 0 : 'R', 1 : 'G'}, indicate the image has the channels
        R and G inpositions 0 and 1 respectivelly.

    display: bool
        if True, the results are plotted.

    cmap: colormap
        colormap to be used in the colorbars and plots.

    Returns
    -------
    sim: 3d array
        Smoothed image
        if the input image is monochromatic, it is not returned.

    sim2d: 2d array
        smoothing applied to the sum of the channels.

    nsim2d: 2d array
        smoothed monochromatic image renormalized.

    """

    dims = image.shape

    # in case is a multiple channels image (like RGB)
    if len(dims) == 3:

        # convert to monochromatic image (default mode = sum the channels)
        im2d = im_to_1chan(image, channels, mode=mono_mode)

        # smooth the image
        sim2d = gaussian_smooth(im2d, ksize, sigma)
        # sim2d = cv2.GaussianBlur(im2d, (ksize, ksize), sigma)

        # renormalize to cover the full image range
        nsim2d = renormalize(
            sim2d
        )  # renormalized to [0,255] for uint8 data or [0,1] for the rest

        if display:
            # start the figure
            fig, axs = plt.subplots(2, 3, figsize=(4 * 3, 6), layout="constrained")
            fig.suptitle("Smoothed with kernell size = " + str(ksize), fontsize=12)

            axs[0, 0].imshow(sim)
            axs[0, 0].set_title("Smoothed Image")

            imi = axs[0, 1].imshow(sim2d, cmap=cmap)
            axs[0, 1].set_title("Monochromatic")
            fig.colorbar(imi, fraction=0.035)

            imi = axs[0, 2].imshow(nsim2d, cmap=cmap)
            axs[0, 2].set_title("Renormalized Monochromatic")
            fig.colorbar(imi, fraction=0.035)

            # go for each channel
            for i in range(dims[2]):

                # display the images
                imi = axs[1, i + 1].imshow(sim[:, :, i], cmap=cmap)
                axs[1, i + 1].set_title(channels[i] + " channel")
                fig.colorbar(imi, fraction=0.035)

        return (sim, sim2d, nsim2d)

    # in case is a monochromatic image
    else:

        ## perform the smoothing
        sim2d = gaussian_smooth(image, ksize, sigma)
        # sim2d = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        nsim2d = renormalize(sim)  # renormalized to [0,1] or [0,255]

        """
        # Renormalize if neccesary
        nSIm = None
        
        if SIm.dtype == np.uint8 and SIm.max() != 255:
            nSIm = renormalize(SIm)
        
        if (SIm.dtype == np.float64 or SIm.dtype == np.float32) and SIm.max() != 1:
            nSIm = renormalize(SIm)
        """
        if display:
            # display the images
            fig, axs = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
            fig.suptitle("Smoothed with kernell size = " + str(ksize), fontsize=12)

            imi = axs[0].imshow(sim2d, cmap=cmap)
            axs[0].set_title("Monochromatic")
            fig.colorbar(imi, fraction=0.035)

            imi = axs[1].imshow(nsim2d, cmap=cmap)
            axs[1].set_title("Monochromatic Renormalized ")
            fig.colorbar(imi, fraction=0.035)

        return (sim2d, nsim2d)


def interactive_smooth(
    image,
    k_range,
    sigma=0,
    mono_mode="sum",
    channels=CHANNELS,
    t_row=None,
    renorm=False,
    cmap="viridis",
    subdir="",
):
    """
    Apply a gaussian smoothing filter to the input image and display it if indicated.
    if multichromatic, it also peform the smoothing over the sum of the channels
    and return this data together its renormalized version.

    The smooth strength is equal for all the channels in multichromatic images
    (this means that channels with lower range of intensity will experiment a
    relative high smooth, which is fine in most cases as we assume noise range
    is equal for all of them)

    *** Por ahora solo esta diseñada para que la imagen sea monocromática

    Parameters
    ----------
    image: array like
         2D or 3D image
         ** Por ahora solo esta diseñada para 2D **

    k_range: list of int
        They must be odd integer numbers with the beggining and end of the kernell values to test
        This is the size of the gaussian kernell.
        typical values range between 5 and 15. The smoothing increase with this value.
        e.g. k_range = [5,15] will considerar kernell sizes of [5,7,9,11,13,15]

    sigma: double
        Filter parameter (standard deviation).
        If sigma = 0, it is automatically determined according the ksize value.
        sigma = 0.3*((ksize-1)/2 -1)+0.8

    mono_mode: string
        mode to convert the image to monochromatic.
        options: 'sum', 'mean', 'renormalize'

    channels: dict
        dictionary with the channels positions and names.
        e.g. channels = { 0 : 'R', 1 : 'G'}, indicate the image has the channels
        R and G inpositions 0 and 1 respectivelly.

    t_row: int
        initial transect row value

    renorm: False
        if True, image is renormalized to cover full data range
        e.g. if data values are between [5,240] and 8bit images admit
        [0,255], the data is ponderated by a factor to cover that range.
        (it increase contrast)

    display: bool
        if True, the results are plotted.

    cmap: colormap
        colormap to be used in the colorbars and plots.

    Returns
    -------
    sim: 3d array
        Smoothed image
        if the input image is monochromatic, it is not returned.

    sim2d: 2d array
        smoothing applied to the sum of the channels.

    nsim2d: 2d array
        smoothed monochromatic image renormalized.

    """
    # init the list to store the image and smoothed images
    ims = [image]
    nims = [renormalize(image)]  # renormalized to [0,1] or [0,255]

    # create the kernell size array based on the input limits
    ksize = np.arange(k_range[0], k_range[1] + 2, 2)

    # init the result dictionary (no se bien que incluir aqui aun)

    result = {
        "smooth_im": None,
        "ks": ksize[0],  # use the fist kernell size as initial value
        "renormalize": renorm,
    }

    # perform a smoothing for each indicated kernell size
    print(f"missing in image {missing_intensities(image)}")

    for ks in ksize:
        print(f"\nstarting the smooth k = {ks} ...")
        sim = gaussian_smooth(image, ks, sigma)
        # sim = cv2.GaussianBlur(image, (ks, ks), sigma)
        print("smooth ready\n")
        print(f"missing in smooth {missing_intensities(sim)}")
        # get the renormalized image too
        nsim = renormalize(sim)  # renormalized to [0,1] or [0,255]
        print(f"missing in renorm {missing_intensities(nsim)}")
        # append the result image to the list
        ims.append(sim)
        nims.append(nsim)

    def get_ssim(ims):  # , renorm):

        ssim_values = [1]  # the is SSIM with itself

        # go for each smoothed image
        for i in range(1, len(ims)):  # don't take the 0, because is the original image

            simi = ims[i]  # smoothed image i

            # if renorm:
            #    ims[i] = renormalize(sim) # renormalized to [0,1] or [0,255]

            # get an score of the smoothing
            ssim_score = calculate_ssim(ims[0], simi)
            ssim_values.append(ssim_score)

        return ssim_values

    # to get the std deviation of the noise rectangle
    def get_std_noise(images, y, x, h, w):

        std_values = []

        for im in images:
            # background image area and compute its std
            bkgn_rectangle = im[y : y + h + 1, x : x + w + 1]  # +1 because the slicing
            bkgn_std = np.std(bkgn_rectangle)

            # append the value
            std_values.append(bkgn_std)

        return std_values

    # create a x vector of the same length as signal
    # x = np.arange(len(signal_vector))
    h_max, w_max = image.shape[0:2]

    # initial background area reactangle
    y0, x0 = int(h_max / 2), int(w_max / 2)
    h0, w0 = int(h_max / 10), int(w_max / 10)

    # get initial transect row position
    if t_row == None or t_row > h_max:

        t_row = int(h_max / 2)

    else:
        # make sure is integer
        t_row = round_up(t_row)

    # create a serie with the position ID of each image
    x_ims_pos = np.arange(1, len(ksize) + 2)  # add one for the image

    ######################
    # Crear la figura una sola vez (plot the signal just to init the lines)
    fig = plt.figure(figsize=(14, 14), layout="constrained")

    # Define a custom axes distribution
    axs = fig.subplot_mosaic(
        [
            ["imO", "imO", "imO", "imO", "0", "imS", "imS", "imS", "imS"],
            ["imO", "imO", "imO", "imO", "0", "imS", "imS", "imS", "imS"],
            ["imO", "imO", "imO", "imO", "0", "imS", "imS", "imS", "imS"],
            ["imO", "imO", "imO", "imO", "0", "imS", "imS", "imS", "imS"],
            ["1", "1", "1", "1", "1", "1", "1", "1", "1"],
            ["metrics", "metrics", "metrics", "metrics", "metrics", "2", "2", "2", "2"],
            ["metrics", "metrics", "metrics", "metrics", "metrics", "2", "2", "2", "2"],
            ["metrics", "metrics", "metrics", "metrics", "metrics", "2", "2", "2", "2"],
            ["3", "3", "3", "3", "3", "3", "3", "3", "3"],
            [
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
            ],
            [
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
            ],
            [
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
            ],
            [
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
            ],
            [
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
                "trans",
            ],
        ]
    )

    # empty figure spaces (they re added to left some space for titles and cbars)
    axs["0"].axis("off")
    axs["1"].axis("off")
    axs["2"].axis("off")
    axs["3"].axis("off")

    ###########################
    # display the original image
    img0 = axs["imO"].imshow(image, cmap=cmap)  # , cmap='gray')
    axs["imO"].set_title("Original image")
    fig.colorbar(
        img0,
        ax=axs["imO"],
        fraction=0.035,
        orientation="vertical",
        label=f"values range: [{min(ims[0].flatten())},{max(ims[0].flatten())}]",
    )
    # dibujar el rectagulo del background
    # ([x0,y0], width, heigh,...)
    rect = Rectangle(
        (x0, y0),
        w0,
        h0,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
        label="background",
    )
    axs["imO"].add_patch(rect)

    # dibujar el transecto tambien en axs[0,0]
    transect = axs["imO"].axhline(
        y=t_row, linewidth=1, color="r"
    )  # trans_line = Rectangle((0,t_row), w_max, 0, linewidth=1, edgecolor='r', facecolor='none', label = 'zoom')
    # add_patch(trans_line)

    #############################
    # display the smoothed image
    ks_pos = (
        np.where(ksize == result["ks"])[0][0] + 1
    )  # add 1 because the 0 is original image
    # in case ksize list --> ks_pos = ksize.index(result['ks']) + 1  # add 1 because the 0 is original image

    img1 = axs["imS"].imshow(ims[ks_pos], cmap=cmap)  # , cmap='gray')
    axs["imS"].set_title(f"Smoothed with kernell size = {result['ks']}")
    cbar = fig.colorbar(
        img1,
        ax=axs["imS"],
        fraction=0.035,
        orientation="vertical",
        label=f"values range: [{min(ims[ks_pos].flatten())},{max(ims[ks_pos].flatten())}]",
    )
    axs["imS"].axis("off")

    #################################
    # el valor del transect en los distintos canales
    # transecto imagen original
    (l0,) = axs["trans"].plot(image[t_row, :], color="k", label="original image")

    # transecto imagen suavizada seleccionada (en base al ks)
    (ls,) = axs["trans"].plot(
        ims[ks_pos][t_row, :], color="r", label=f"smooth ks={result['ks']}"
    )

    axs["trans"].legend(handles=[l0, ls])

    axs["trans"].set_title(f"transect in row {t_row}", pad=-20)

    ###############################
    # el resultado de las SSIM
    ssim_values = get_ssim(ims)  # , renorm)
    ssim_values_norm = get_ssim(nims)  # , renorm)

    (lss,) = axs["metrics"].plot(
        x_ims_pos, ssim_values, marker="x", color="k", label="SSIM w/o renorm"
    )
    (lrss,) = axs["metrics"].plot(
        x_ims_pos, ssim_values_norm, marker=".", color="tab:gray", label="SSIM renorm"
    )

    smooth_text = ["original"]
    smooth_text.extend(ksize)
    axs["metrics"].set_xticks(x_ims_pos, smooth_text)

    axs["metrics"].set_xlabel("k_size")
    axs["metrics"].set_ylabel("Structural Similarity Score")

    axs["metrics"].set_title("SSIM and STD of background (noise)", pad=-20)

    # the standard deviation of noise
    noise_std = get_std_noise(ims, y0, x0, h0, w0)
    noise_std_norm = get_std_noise(nims, y0, x0, h0, w0)

    ax_std = axs["metrics"].twinx()  # to share the same x-axis
    (ln,) = ax_std.plot(
        x_ims_pos, noise_std, marker="x", color="tab:red", label="std w/o renorm"
    )
    (lrn,) = ax_std.plot(
        x_ims_pos, noise_std_norm, marker=".", color="tab:orange", label="std renorm"
    )

    ax_std.set_ylabel("Standard deviation")  # , color = 'tab:red')

    # the ratio of both as SNR proxy
    # get the maximum values to normalize the STD
    max_noise = max(noise_std)
    max_noise_norm = max(noise_std_norm)

    ssim_noise = [i / (j / max_noise) for i, j in zip(ssim_values, noise_std)]
    ssim_noise_norm = [
        i / (j / max_noise_norm) for i, j in zip(ssim_values_norm, noise_std_norm)
    ]

    ax_ratio = axs["metrics"].twinx()  # to share the same x-axis
    (l_ratio,) = ax_ratio.plot(
        x_ims_pos, ssim_noise, marker="x", color="tab:blue", label="ratio w/o renorm"
    )
    (lrn_ratio,) = ax_ratio.plot(
        x_ims_pos, ssim_noise_norm, marker=".", color="tab:cyan", label="ratio renorm"
    )

    ax_ratio.set_ylabel("SNR (ratio SSIM/STD%)")  # , color = 'tab:red')

    # move the spine of the second axes outwards
    ax_ratio.spines["right"].set_position(("axes", 1.15))

    # set the color of the y-axis and y-axis text
    axs["metrics"].yaxis.label.set_color(lss.get_color())
    ax_std.yaxis.label.set_color(ln.get_color())
    ax_ratio.yaxis.label.set_color(l_ratio.get_color())

    axs["metrics"].tick_params(axis="y", labelcolor=lss.get_color())
    ax_std.tick_params(axis="y", labelcolor=ln.get_color())
    ax_ratio.tick_params(axis="y", labelcolor=l_ratio.get_color())

    axs["metrics"].legend(
        handles=[lss, lrss, ln, lrn, l_ratio, lrn_ratio],
        bbox_to_anchor=(1.3, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    # Eliminar la figura previa a los widget, que queda como un duplicado congelado.
    plt.close(fig)

    # Function to update the smoothed image
    def update_image(ks, renorm):

        ks_pos = (
            np.where(ksize == ks)[0][0] + 1
        )  # add 1 because the 0 is original image
        # in case ksize list --> ks_pos = ksize.index(result['ks']) + 1  # add 1 because the 0 is original image

        if renorm:
            current_im = nims[ks_pos]
            axs["imS"].set_title(f"Smoothed with kernell size = {ks}, renormalized")
        else:
            current_im = ims[ks_pos]
            axs["imS"].set_title(f"Smoothed with kernell size = {ks}")

        img1 = axs["imS"].imshow(current_im, cmap=cmap)  # , cmap='gray')

        # update the colorbar limits accrod the current image (I don't use same for all to see the current contrast clearly)
        cbar.mappable = img1
        cbar.update_normal(img1)  # limits are normalized to the current image

        cbar.set_label(
            f"values range: [{min(current_im.flatten())},{max(current_im.flatten())}]"
        )

        with output:
            # store the values
            result["ks"] = ks
            result["renormalize"] = renorm
            result["smooth_im"] = current_im

            # display the figure
            clear_output(wait=True)
            display(fig)

    # Function to update the transect
    def update_transect(row, renorm):

        # update the transect horizontal line in image
        transect.set_ydata([row, row])

        # upsate the values
        ks_pos = (
            np.where(ksize == result["ks"])[0][0] + 1
        )  # add 1 because the 0 is original image

        if renorm:

            t_im = nims[0][row, :]
            t_ksim = nims[ks_pos][row, :]
            label_O = "original image renorm"
            label_ks = f"smooth ks={result['ks']}, renorm"

        else:
            t_im = ims[0][row, :]
            t_ksim = ims[ks_pos][row, :]
            label_O = "original image"
            label_ks = f"smooth ks={result['ks']}"

        # update the lines
        l0.set_ydata(t_im)
        l0.set_label(label_O)
        ls.set_ydata(t_ksim)
        ls.set_label(label_ks)

        axs["trans"].legend(handles=[l0, ls])  # loc = 'upper left')

        axs["trans"].set_title(f"transect in row {row}", pad=-20)

        # Redibujar la figura sin volver a renderizar todo
        with output:
            # store the values
            # result['renormalize'] = renorm

            # display the figure
            clear_output(wait=True)
            display(fig)

    # Function to update the background noise
    def update_rectangle(y, x, h, w):
        # update the rectangle in ax[0,0] and the std values in ax[1,1]

        ##### update the rectangle #####
        rect.xy = (x, y)
        rect.set_height(h)
        rect.set_width(w)

        #### the standard deviation of noise #####
        noise_std = get_std_noise(ims, y, x, h, w)
        noise_std_norm = get_std_noise(nims, y, x, h, w)

        # update the lines
        ln.set_ydata(noise_std)
        lrn.set_ydata(noise_std_norm)

        # update the axis limits
        ymin = min(min(noise_std), min(noise_std_norm))
        ymax = max(max(noise_std), max(noise_std_norm))
        y_range = ymax - ymin
        # give some relaxation accorded to the range
        ymin = ymin - 0.05 * y_range
        ymax = ymax + 0.05 * y_range

        ax_std.set_ylim(ymin, ymax)

        #### the ratio SSIM/STD ####
        max_noise = max(noise_std)
        max_noise_norm = max(noise_std_norm)

        ssim_noise = [i / (j / max_noise) for i, j in zip(ssim_values, noise_std)]
        ssim_noise_norm = [
            i / (j / max_noise) for i, j in zip(ssim_values_norm, noise_std_norm)
        ]

        # update the lines
        l_ratio.set_ydata(ssim_noise)
        lrn_ratio.set_ydata(ssim_noise_norm)

        # update the axis limits
        ymin_r = min(min(ssim_noise), min(ssim_noise_norm))
        ymax_r = max(max(ssim_noise), max(ssim_noise_norm))
        y_range_r = ymax_r - ymin_r

        # give some relaxation accorded to the range
        ymin_r = ymin_r - 0.05 * y_range_r
        ymax_r = ymax_r + 0.05 * y_range_r

        ax_ratio.set_ylim(ymin_r, ymax_r)

        # Redibujar la figura sin volver a renderizar todo
        with output:
            # store the values
            # result['renormalize'] = renorm

            # display the figure
            clear_output(wait=True)
            display(fig)

    """
    # proceed accord the image dimensions
    dims = image.shape
    
    # in case is a multiple channels image (like RGB)
    if len(dims) == 3:
        
        # convert to monochromatic image
        im2d = im_to_1chan(image, channels, mode = mono_mode)
        
        # smooth the image
        sim2d = cv2.GaussianBlur(im2d, (ksize, ksize), sigma)
        
        #renormalize to cover the full image range
        nsim2d = renormalize(sim2d)     # renormalized to [0,255] for uint8 data or [0,1] for the rest
        
        if display:
            # start the figure
            fig, axs = plt.subplots(2, 3, figsize=(4*3, 6), layout='constrained')
            fig.suptitle('Smoothed with kernell size = '+str(ksize), fontsize=12)
            
            axs[0,0].imshow(sim)
            axs[0,0].set_title('Smoothed Image')
            
            imi = axs[0,1].imshow(sim2d, cmap = cmap)
            axs[0,1].set_title('Monochromatic')
            fig.colorbar(imi, fraction = 0.035)
            
            imi = axs[0,2].imshow(nsim2d, cmap = cmap)
            axs[0,2].set_title('Renormalized Monochromatic')
            fig.colorbar(imi, fraction = 0.035)


            #go for each channel
            for i in range(dims[2]):
                
                # display the images
                imi = axs[1,i+1].imshow(sim[:,:,i], cmap = cmap)
                axs[1,i+1].set_title(channels[i]+' channel')
                fig.colorbar(imi, fraction = 0.035)
                
        return(sim, sim2d, nsim2d)
    
    # in case is a monochromatic image    
    else:
    """

    # Cuadro de texto con instrucciones
    instructions = widgets.HTML(
        value="""
        <h3>Instructions:</h3>
        Get the Structural Similarity score of the images (score between [-1,1], a 1 indicated perfect similarity).
        <ul>
            <li>Select an background area as a proxy of the noise and analyze it</li>
            <li>select/un-select "normalize box" to make smooth image cover full pixel range.</li>
        </ul>
        """,
        layout=widgets.Layout(width="800px"),
    )

    noise_instructions = widgets.HTML(
        value="""
        </ul>
        Modify the background area with the next sliders:
        </ul>
        """,
        layout=widgets.Layout(width="800px"),
    )

    # Cuadro de texto para el nombre del archivo
    filename_input = widgets.Text(
        value="smooth_explore.pdf",
        placeholder="filename",
        description="Filename:",
        continuous_update=False,
        layout=widgets.Layout(width="400px"),
    )

    # Botón para guardar la imagen
    save_button = widgets.Button(
        description="Save Image", layout=widgets.Layout(width="200px"), indent=True
    )

    # Evento del botón de guardado
    def on_save_button_click(_):
        # filename = filename_input.value  # Obtener el nombre del archivo
        # save_figure(filename)
        ofname = (
            filename_input.value
        )  # Obtener el nombre del archivo f'pixel_to_mm_mean'

        """Función para guardar la figura actual."""
        if fig is not None:
            fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
            save_figure(fig, ofname, subdirs=subdir, **fig_kwargs)
            print(f"Imagen guardada como: {ofname}")
        else:
            print("No hay ninguna figura para guardar.")

    save_button.on_click(on_save_button_click)

    # Normalize checkbox
    norm_input = widgets.Checkbox(
        value=result["renormalize"],
        description=" Renormalize the smoothed figure",
        disabled=False,
        indent=False,
    )

    # Creación de los slider
    ks_slider = widgets.IntSlider(
        min=k_range[0],
        max=k_range[1],
        step=2,
        value=result["ks"],
        description="kernel size",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    row_slider = widgets.IntSlider(
        min=0,
        max=h_max - 1,
        step=1,
        value=h_max / 2,
        description="row transect",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    y_slider = widgets.IntSlider(
        min=0,
        max=h_max - 1,
        step=1,
        value=y0,
        description="y0",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    x_slider = widgets.IntSlider(
        min=0,
        max=w_max - 1,
        step=1,
        value=x0,
        description="x0",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    h_slider = widgets.IntSlider(
        min=1,
        max=h_max - 1,
        step=1,
        value=h0,
        description="height",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    w_slider = widgets.IntSlider(
        min=1,
        max=w_max - 1,
        step=1,
        value=w0,
        description="wide",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )

    # Función que se ejecuta al mover ks slider (vinculando la actualización de la figura al slider)
    def on_ks_change(change):
        update_image(ks_slider.value, norm_input.value)
        update_transect(row_slider.value, norm_input.value)

    # Función que se ejecuta al mover el row slider (vinculando la actualización de la figura al slider)
    def on_row_change(change):
        update_transect(row_slider.value, norm_input.value)

    # Función que se ejecuta al mover los square slider (vinculando la actualización de la figura al slider)
    def on_backgnd_change(change):
        update_rectangle(y_slider.value, x_slider.value, h_slider.value, w_slider.value)

    # Función que se ejecuta al presionar renormalize checkbox (vinculando la actualización de la figura al slider)
    def on_renorm_change(change):
        update_image(ks_slider.value, norm_input.value)
        update_transect(row_slider.value, norm_input.value)

    # Conectar un evento en el slider con la función de actualización
    ks_slider.observe(on_ks_change, names="value")
    row_slider.observe(on_row_change, names="value")
    y_slider.observe(on_backgnd_change, names="value")
    x_slider.observe(on_backgnd_change, names="value")
    h_slider.observe(on_backgnd_change, names="value")
    w_slider.observe(on_backgnd_change, names="value")
    norm_input.observe(on_renorm_change, names="value")

    # Controles de widgets
    controls = widgets.VBox(
        [
            instructions,
            widgets.HBox([ks_slider, row_slider]),
            norm_input,
            noise_instructions,
            widgets.HBox([y_slider, x_slider]),
            widgets.HBox([h_slider, w_slider]),
            widgets.HBox([filename_input, save_button]),
        ]
    )

    # es necesario hacer display(output) para que se desplieguen los
    # efectos de manipular los widgets
    output = widgets.Output()

    # Mostrar los widget y la figura inicial
    display(controls, output)

    # Desplegar la figura con los valores iniciales de los widgets.
    update_transect(row_slider.value, norm_input.value)
    update_image(ks_slider.value, norm_input.value)
    update_rectangle(y_slider.value, x_slider.value, h_slider.value, w_slider.value)

    return result


def eliminate_bright(im2d, bright_thresh=0.9, replace_value=0, display=False):
    """
    eliminate monochromatic image brigth pixels over the threshold
    (to eliminate noisi elements?)

    """

    imwob = np.where(image <= bright_thresh, im2d, replace_value)

    if display:
        plt.figure()
        plt.imshow(imwob)
        plt.colorbar()

    return imwob


def scale_im(image, factor, anti_alias=True, chan_dim=None, **kwargs):
    """
    To re-scale an image. It is considered as monochromatic at leat chan_dim is
    indicated (the dimention that indicate the channels, usually the 3rd)

    Parameters
    -----------
    image: array like
        The image array to be rescaled. It could be monochromatic or multichromatic.

    factor: numeric
        The rescaling factor.
        e.g. if factor = 10, the image size is reduced 10 times

    anti_alias: boolean
        if True, anti-aliasing is performed (Gaussian filter)
        to smooth the image prior to re-scaling.
        It is totally recomenden when downscaling

    chan_dim: integer
        if given, this image shape dimention is considered to contain the different
        channels. If None, its considered monochromatic.
        This parameter isn't the number of channels but the image.shape
        dimention that contain that information.

    **kwargs:
        to set other options of the skimage.transform.rescale function.
        e.g. preserve_range = True/False . Default False
        True to keep the original range of values.
        Faslse, the input image is converted according to the conventions of img_as_float

    Returns
    -------
    im_rescaled: array like
        Rescaled image array. 2D if chan_dim = None. n-dim is chans_dim
        is indicated (where is the number of channels)

    """
    text = ""
    if anti_alias:
        text = " (anti aliased)"

    im_rescaled = rescale(
        image, 1 / factor, anti_aliasing=anti_alias, channel_axis=chan_dim, **kwargs
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")

    axes[0].imshow(image)  # , cmap='gray')
    axes[0].set_title("Original image")

    axes[1].imshow(im_rescaled)  # , cmap='gray')
    axes[1].set_title("Recaled 1/" + str(factor) + text)

    return im_rescaled


def calculate_ssim(im, smoothed_im, display_map=False, data_range=None, **kwargs):
    """
    to calculate the Structural Similarity between two images as a measure of
    the smoothing degree.
    It is compulsory that input image data has the same number of channels and shape.

    SSIM measures the structural similarity between two images by comparing
    three aspects: luminance, contrast, and structure. The difference in
    luminance and contrast is evaluated based on the range of pixel intensity
    values, and data_range is used to normalize these differences.

    Parameters:
    -----------
    im: image data (array)
        original image

    smoothed_im: image data (array)
        smoothed image data

    display_map: bool
        if True, plot the structural similarity map

    data_range: float
        "The data range of the input images". This value defines the diffenrece
        between pixels values which is asigned as maximum score. Any bigger
        difference will be asigned the same maximum score, then it should cover
        the whole range of images values.

        By default (data_range = None), is computed as the distance between
        minimum and maximum values across both images.

        This value should cover the full range of images values and be the more
        close to them.
            -if range value is too big, the diferences will be understimated and
            SSIM value will be bigger than it should be.
            - if range value is too small, the diffences will be overstimated
            and SSIM value will be smaller than it should be.

        e.g if actual range of images is 10 to 220, the ideal will be to considerer
        a data_range of 210 (because is the bigger variation between two pixels)
        but if you set data_range = 1000, the differences between pixels that at
        most will be 210, are just a little fraction of the 1000 value.
        if you set data_range = 1, any differene will be considered as maximum diffence.
        (say a difference of 1 or 210 between two pixels will have the same maximum score)

    **kwargs:
        any other optional parameters to pass to skimage.metrics.structural_similarity

    Returns:
    --------
    ssim_score: list
        ssim value for each channel in the same order as channels.
        each valie is a number between [-1,1].
        A value of 1 indicated perfect simmilarity.
    """

    # compute the data range
    im_max = im.max()
    sim_max = smoothed_im.max()
    im_min = im.min()
    sim_min = smoothed_im.min()

    im_range = im_max - im_min
    sim_range = sim_max - sim_min

    # check they are similar
    range_ratio = max(im_range, sim_range) / min(im_range, sim_range)

    if range_ratio > 2:
        print(
            "\nWARNING: range of image values is quite different and SSIM will be "
            "imprecise in luminance and contraste differences consideration\n"
        )
        print(f"range image: {im_range}")
        print(f"range smoothed image: {sim_range}\n")

    # if data range was not defined use previous values to define it
    if data_range == None:

        data_range = max(im_max, sim_max) - min(im_min, sim_min)

    # For monochromatic images
    if im.ndim == 2 and smoothed_im.ndim == 2:

        ssim_score, ssim_map = ssim(
            im, smoothed_im, full=True, data_range=data_range, **kwargs
        )

        # dado que full = True, devuelve el mapa de SSIM, el cual se puede graficar:
        if display_map:
            plt.imshow(ssim_map, cmap="gray")
            plt.colorbar()
            plt.show()

    # For multichannels
    elif im.ndim == 3 and smoothed_im.ndim == 3:

        # init the lists to store the values
        ssim_score = list()
        ssim_map = list()

        nchan = im.shape[2]  # number of channels

        for i in range(nchan):

            score_i, map_i = ssim(
                im[:, :, i],
                smoothed_im[:, :, i],
                full=True,
                channel_axis=2,
                data_range=data_range,
                **kwargs,
            )
            ssim_score.append(score_i)
            ssim_map.append(map_i)

        if display_map:

            fig, axs = plt.subplots(1, nchan, figsize=(4 * nchan, 2.5))

            for i in range(nchan):

                axs[i].imshow(ssim_map[i], cmap="gray")
                axs[i].colorbar()

            plt.show()

    else:
        raise Exception("Number of channels are not the same for the input images")

    return ssim_score


def eval_smooth(image, fim, row, data_range=None, normalize=False, vlims=False):
    """
    To visualize and evaluate the smoothing.
    It display a transect over the original image and over the smoothed image as a visual
    aproximation of the obtained change (to get a good smooth but not an oversmooth)

    #pendientes:
      - Hacer que acepte hacer transectos en columnas tambien (si se ingresa, que arroje para ese también la serie de plots)
      - Chequear lo de la normalización

    image: image array
        it could be a (m,n,k) array, where k is optional.

    fim: filtered image array
        Image array obtained after the smoothing proccess.

    row: int
        transect position to visualize the result

    data_range: float
        To use in the Structural Similarity computation.
        "The data range of the input image
        (distance between minimum and maximum possible values).
        By default, this is estimated from the image data type.
        This estimate may be wrong for floating-point image data.
        Therefore it is recommended to always pass this value explicitly "


    normalize: True or numeric
        If True, the original image is renormalized to [0,1]
        if numeric values is given, it is used to nomarlize the data,
        dividing it by this value.

    vlims: list
        y axis limits in transect plot.
        e.g. [10, 500]

    """
    # Normalization
    # nfim = (fim-fim.min())/(fim.max()-fim.min())

    # convert range of image values to [0,1] if indicated
    try:
        if normalize:
            image = renormalize(
                image, factor=1, dtype="float64"
            )  # nim = (image-image.min())/(image.max()-image.min())
    except:
        # "manual" normalization
        if isinstance(normalize, (int, float)):
            image = image / normalize

    # get an score of the smoothing
    ssim_score = calculate_ssim(image, fim, data_range=data_range)
    print(f"\nThe Structural Similarity score of the images is: {ssim_score}")
    print("(score between [-1,1], a 1 indicated perfect similarity)")
    #
    print("Transect at " + str(row) + " pixel in y-axis")

    # check the dimentions
    h1, w1 = image.shape[0:2]
    h2, w2 = fim.shape[0:2]

    row2 = row

    if h1 != h2:
        row2 = int(row * (h2 / h1))  # rescale the row to the heigh of fim

    dsize = False
    if w1 != w2:
        dsize = True  # different size

    # plot the smoothed and original image with the transect line

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axhline(row, color="r", ls="--", label="transect")
    plt.title("original image")

    plt.subplot(1, 2, 2)
    plt.imshow(fim)  # or nfim
    plt.axhline(row2, color="r", ls="--", label="transect")
    plt.title("smoothed image")

    # plot the transect values previous and after the smoothing

    x1 = [i for i in range(0, w1)]
    x2 = [i for i in range(0, w2)]

    # get the number of channels
    try:
        nc = image.shape[2]
    except:
        nc = 1

    # if there are more than 1 channel
    if nc > 1:

        fig, axs = plt.subplots(
            1, nc, figsize=(14, 3), layout="constrained"
        )  # iniciamos la figura
        fig.suptitle("Transect value", fontsize=12)  # y=0.95)
        # plt.gcf().suptitle("Transect value")

        for i in range(nc):
            # plt.subplot(1, nc, i+1)
            (l1,) = axs[i].plot(x1, image[row, :, i], color="k", label="original")
            axs[i].set_title("Canal " + str(i + 1))

            if dsize:
                axi2 = axs[
                    i
                ].twiny()  # instantiate a second axes that shares the same y-axis
                (l2,) = axi2.plot(x2, fim[row2, :, i], color="r", label="smoothed")
                # axi2.set_xlabel('fim_wide', color='r')  # we already handled the x-label with ax1
                # axi2.tick_params(axis='x', labelcolor='r')

            else:
                (l2,) = axs[i].plot(x2, fim[row2, :, i], color="r", label="smoothed")

            # add the legent to the last ax
            if i == nc - 1:
                axs[i].legend(handles=[l1, l2])

            if type(vlims) != bool:
                try:
                    axs[i].set_ylim(vlims[0], vlims[1])

                except:
                    print("vlims input invalid")
    # if it's a monochromatic image
    else:
        fig, ax = plt.subplots(
            layout="constrained"
        )  # (figsize=(14,3), layout='constrained') # iniciamos la figura
        (l1,) = ax.plot(x1, image[row, :], color="k", label="original")
        ax.set_title("Transect value")

        if dsize:
            ax2 = ax.twiny()
            (l2,) = ax2.plot(x2, fim[row2, :], color="r", label="smoothed")

        else:
            (l2,) = ax.plot(x2, fim[row2, :], color="r", label="smoothed")

        ax.legend(handles=[l1, l2])

        if type(vlims) != bool:
            try:
                ax.set_ylim(vlims[0], vlims[1])

            except:
                print("vlims input invalid")


def image_shift(
    image,
    dy,
    dx,
    method_name="INTER_CUBIC",
    mode_name="BORDER_CONSTANT",
    b_value=0,
    show_fig=False,
    title="",
):
    """
    Shift the input monochromatic image data in the indicated dy and dx values
    (they are notrestricted to integers) using cv2.warpAffine() function.

    ** add some smart methods to determine b_value (like the mean of 5% lower values) **

    Parameters:
    -----------
    image : np.ndarray
        Monochromatic image of dimentions (N, M).

    dy : numeric
        y-axis displacement

    dx : numeric
        x-axis displacement

    method_name: string
        Name of the method used to interpolate the image values in case
        non-integer displacement is applied.

    mode_name: string
        Name of the mode used to fill the shifted borders.

    b_value: numeric or string
        border value used to fill the shifter borders when 'BORDER_CONSTANT'
        is applied.
        ** there is no String method implemented yet **

    show_fig: bool
        if True, the original and shift images are displayed

    title: str
        string to add to the Original Image title

    Returns:
    --------
    shift_image: np.ndarray
        shifted image accord the inputs

    """
    # dicts with possible methods and modes
    methods = {
        "INTER_NEAREST": cv2.INTER_NEAREST,
        "INTER_LINEAR": cv2.INTER_LINEAR,
        "INTER_CUBIC": cv2.INTER_CUBIC,
        "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
    }

    modes = {
        "BORDER_CONSTANT": cv2.BORDER_CONSTANT,
        "BORDER_REPLICATE": cv2.BORDER_REPLICATE,
        "BORDER_REFLECT": cv2.BORDER_REFLECT,
        "BORDER_REFLECT_101": cv2.BORDER_REFLECT_101,
        "BORDER_WRAP": cv2.BORDER_WRAP,
    }

    # get the indicated
    method = methods[method_name]
    mode = modes[mode_name]

    # transformation matrix for the displacement
    M_transform = np.float32([[1, 0, dx], [0, 1, dy]])

    # get image shape
    N, M = image.shape

    # Perform the shift
    if mode == cv2.BORDER_CONSTANT:
        shift_image = cv2.warpAffine(
            image,
            M_transform,
            (M, N),
            flags=method,
            borderMode=mode,
            borderValue=b_value,
        )
    else:
        shift_image = cv2.warpAffine(
            image, M_transform, (M, N), flags=method, borderMode=mode
        )

    # Display the results
    if show_fig:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(image)  # , cmap='gray')
        axs[0].set_title(f"Original{title}")
        axs[1].imshow(shift_image)  # , cmap='gray')
        axs[1].set_title(f"{method_name} - {mode_name}")
        plt.show()

    return shift_image


def data_shift(
    data,
    channels,
    dy_spline,
    dx_spline,
    frames_spline=[],
    subtract=False,
    method_name="INTER_CUBIC",
    mode_name="BORDER_CONSTANT",
    b_value=0,
    show_frames=[],
    int_shift=False,
):
    """
    Shift the images in data as indicated by the dy and dx splines (they are not
    restricted to integers) using cv2.warpAffine() function.
    This splines contain the shift value related to each frame.

    By default the splines are evaluated using the frames index, but if something
    different was used to generate them you can input a custom vector to evaluate
    the splines ("frames_spline" argument)

    By default the images shift is not restricted to integers and a iterpolation
    is performed if they arent (using the method indicated in method_name)
    If interpolation is not desired, you can force the shift values to be integers
    with "int_shift = True". 'INTER_NEAREST' interpolation method should be equivalent.

    See cv2.warpAffine() documentation for detailed information about the
    interpolation methods and border filling modes.

    Parameters:
    -----------
    data : dictionary
        each element is a channel with image data as a np.ndarray of (N,M,L)
        dimentios, where L is the number of frames. Example:
        data['G'] = [N,M,L]

    channels: list
        list with the data channels keys chosen to be shift.

    dy_spline : spline object
        y-axis displacement spline (class 'scipy.interpolate._fitpack2.UnivariateSpline')

    dx_spline : spline object
        x-axis displacement spline (class 'scipy.interpolate._fitpack2.UnivariateSpline')

    frames_spline: list or 1d-array
        vector used to evaluate the splines. By default the splines are evaluated
        using the frames index.

    subtract: boolean
        True to subtract the input displacement.
        False by default.

    method_name: string
        Name of the method used to interpolate the image values in case
        non-integer displacement is applied. Options:
        {'INTER_NEAREST','INTER_LINEAR','INTER_CUBIC','INTER_LANCZOS4'}

    mode_name: string
        Name of the mode used to fill the shifted borders. Options:
        {'BORDER_CONSTANT','BORDER_REPLICATE','BORDER_REFLECT',
        'BORDER_REFLECT_101','BORDER_WRAP'}

    b_value: numeric or string
        border value used to fill the shifter borders when 'BORDER_CONSTANT'
        is applied.
        ** there is no String method implemented yet **

    show_frames: list of integers
        List with the index of image frames to display.
        By default any image is displayed.

    int_shift: boolean
        If True, the x and y shift values are converted to integers with
        roud_up() function. By default is False.

    Returns:
    --------
    shifted_data: dictionary with nd.arrays
        dictionary with the indicated channels data shifted accord the input
        splines and options.
        Structure: [channels][N,M,L]
    """

    # init the output dictionary
    shifted_data = {}

    for chan in channels:
        # get the channel data and shape
        datac = data[chan]
        N, M, L = datac.shape

        # init the channel shifted data array
        shifted_data[chan] = np.zeros_like(datac)

        # evaluate the splines
        if len(frames_spline) != L:

            if len(frames_spline) != 0:
                # give some warning
                print(
                    f'\n"frames_spline" is not the same lenght as channel "{chan}" frames.'
                )
                print(
                    f"splines will be evaluated by the frames index in the data array.\n"
                )

            frames_spline = np.arange(L)

        # evaluate the shift inputs in the indicad values of frames_spline
        shifts = [dy_spline, dx_spline]

        for i in range(len(shifts)):

            # if input is a list, convert to array
            if type(shifts[i]) is list:
                shifts[i] = np.asarray(shifts[i])

            # if array, get the index indicated in frame_splines
            if type(shifts[i]) is np.ndarray:
                shifts[i] = shifts[i][frames_spline]

            # if are splines (or other function), evaluate them in frame_splines
            else:
                shifts[i] = shifts[i](frames_spline)

        shift_y = shifts[0]
        shift_x = shifts[1]

        # in case discretize the values was indicated
        if int_shift:
            shift_y = round_up(shift_y)
            shift_x = round_up(shift_x)

        # in case subtract the input values was indicated
        if subtract:
            shift_y = -shift_y
            shift_x = -shift_x

        # go for each frame
        for i in range(L):

            # get the frame shift values
            dy = shift_y[i]
            dx = shift_x[i]

            image = datac[:, :, i]

            # display the shift image if indicated
            if i in show_frames:
                display_frame = True
                add_title = f" {i}"
            else:
                display_frame = False
                add_title = ""

            # Perform the shift
            shifted = image_shift(
                image,
                dy,
                dx,
                method_name,
                mode_name,
                b_value,
                show_fig=display_frame,
                title=add_title,
            )

            # shifted = image_shift(
            #    image,
            #    dy,
            #    dx,
            #    method_name,
            #    mode_name,
            #    b_value,
            #    show_fig=False,
            #    title=add_title,
            # )

            shifted_data[chan][:, :, i] = shifted

        print(f"\nChannel {chan} shifted\n")

    return shifted_data, [shift_y, shift_x]


def moving_average(im2d, wsize, cmap=None):
    """
    im2D:  array like
    2D image array

    wsize: tuple
        rolling windows size. It has to be same number of dimentions as im2D
        e.g. = (2,2)
    """
    # define the colormap to be used
    try:
        if cmap == None:
            matplotlib.colormaps.get_cmap("viridis")
    except:
        pass

    # compute the moving average
    win = np.ones(wsize)
    mavg = signal.convolve(im2d, win, mode="same") / sum(sum(win))  # moving_average

    # plot the results
    fig, axs = plt.subplots(1, 2, figsize=(17, 6), layout="constrained")
    imi = axs[0].imshow(im2d, cmap=cmap)
    axs[0].set_title("original")
    fig.colorbar(imi, fraction=0.035)

    imi = axs[1].imshow(mavg, cmap=cmap)
    axs[1].set_title("Moving average")
    fig.colorbar(imi, fraction=0.035)

    return mavg


def background_variation(
    splines, n_frames, channel, initial_ids, spline_key="spline", display=False
):
    """
    To obtain the mean background variation of the input splines around the
    mean value in the initial images frames.


    Parameters
    ----------
    splines: list
        List of dictionaries obtained with the flup.inspect_signal() function
        Each dictionary contains a spline object asociated to [spline_key].

    n_frames: integer
        Number of frames to build the background variation vector.
        The splines are evaluated in the space defined by that number of frames.

    initial_ids : list or array of integers
        ids asociated to the initial images, the ids previously used to
        compute the average background image.

    spline_key: string
        dictionary key asociated to spline object of input "splines"

    channel: str
        channel name used to create the output dictionary


    Return
    ------
    mean_spline: dictionary
        mean_spline[channel] contains the result mean spline of input splines
        evaluated in the n_frames range

    mean_discretized: np.array
        discretized values of mean_splines by round them to the nearest integer.

    """

    # convert splines to a list
    if type(splines) != list:
        splines = [splines]

    ###### process and plot them ######
    # Eval the splines in each frame and get the pixel variation relative to the first times
    x_spline = np.arange(n_frames)
    nbgs = list()  # create a list for the "normalized backgound variation over time"

    for sp in splines:
        # eval the spline in the frames
        sp_x = sp[spline_key](x_spline)

        # normalization
        # compute the pixel variation around the mean value of the initial frames
        nbg = sp_x - sp_x[initial_ids].mean()  # max(bg1_spline['spline'](x_spline))
        nbgs.append(nbg)

    # get the mean of them
    mean_spline = {channel: None}  # init it as a dictionary
    mean_spline[channel] = np.asarray(nbgs).mean(axis=0)
    mean_discretized = round_up(mean_spline[channel])

    if display:
        # display them
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        # plot the splines
        for i, (sp, col_i) in enumerate(zip(nbgs, ["y", "c", "g"])):
            ax.plot(x_spline, sp, col_i, marker="", linestyle=":", label=f"empty {i}")

        ax.plot(x_spline, mean_spline[channel], "k:", label="mean")
        ax.plot(x_spline, mean_discretized, "k", label="discretized")
        ax.legend()
        ax.set_xlabel("Frame number")
        ax.set_ylabel("Normalized mean pixel variation")

        ax.set_ylim([mean_discretized.min() - 1, mean_discretized.max() + 1])
        ax.set_title("Background variation over time")
        plt.show()

    return (mean_spline, mean_discretized)


def colony_blobs_id(
    im2d,
    thresh,
    sigma_lims=[1, 10],
    nsigma=-1,
    max_over=0.8,
    max_val=255,
    text_displace=[-1, -1],
    display=True,
    show_ids=True,
    ids_color="white",
    exclude_border=False,
    filter_circle=None,
    **kwargs,
):
    """
    Use skimage to identify the center position and radius of each colony.
    It recognize colonies as bright blobs in a dark background.

    It also includes optional display of the blobs over the image.
    If you want to set more parameters on the image (like zoom, line colors, etc)
    use **kwargs or the flup.plot_blobs() function with the output blobs.

    Parameters
    ----------
    im2d: array
        array of single channel image data, it is, shape MxN.

    thresh:
        Pixel values > thresh are included in the analysis.
        As blob log use the image in the [0,1] space, theshold has to be
        normalized to that space too. In case it is bigger than 1 its value
        is normalized by max_val parameter.

    sigma_lims: list [min,max]
        Indicates the minimum and maximum sigma to search for colonies.
        The actual radio of a colony is: sqrt(2)*sigma

    max_over: int or float
        Indicates the maximum overlap allowed between two colonies.
        If the area of two colonies overlaps by a fraction greater than threshold,
        the smaller colony is con taked in account.

    nsigma: integer
        number of intermediate sigma values between the sigma limits to test.

    max_val: numeric
        maximum value of the scale of image data. This value is used to normalize
        threshold to be consitent with the [0,1] value space used by blob.log()

    display: bool
        if True, display the blobs in the input image

    show_ids: bool
        if True, display the IDs (i.e. position in the blobs array) of
        each blob together its plotted circle.

    text_displace: list with two numbers
        the factor applied to displace the text from the circles to the upper left image corner.
        This factor is multiplied by the radius of each colony.
        e.g. text_displace = [2,1] --> text will be displaced [-2*ri, ri],
        where ri is the radius of colony i.

    ids_color: any matplotlib color indicator
        color used to display the blobds ids over the image

    filter_circle: dictionary
        It contains the values of the circular area which is used to filter
        the found blobs. Then, valid colonies are restricted to those who has
        its center inside this circular area.
        e.g. filter_circle = {'y0,x0': [1301, 1629], 'r': 745}

    **kwargs:
        to use in flup.plot_blobs() function

    Returns
    -------
    blobs: array (Nx3)
        contains the (y,x) center position and radius of each blob
        for each of N colonies detected in the im2d data.
        estructure for the 'i' element: [yi, xi, ri]
        r is calculated as (2**0.5)*sigma

    """
    # in case it was not specified use the sigma limits to define unitary steps values
    if nsigma == -1:
        nsigma = int(sigma_lims[1] - sigma_lims[0])

        # but use a minimum of 10
        if nsigma < 10:
            nsigma = 10

    if thresh > 1:
        thresh = thresh / max_val

    # perform the detection (divide im2d by 1 to convert to float for skfeat propper processing)
    blobs = skfeat.blob_log(
        im2d.astype("float"),
        min_sigma=sigma_lims[0],
        max_sigma=sigma_lims[1],
        num_sigma=nsigma,
        threshold=thresh,
        overlap=max_over,
        exclude_border=exclude_border,
    )

    # filter the blobs that are inside the circle
    if filter_circle != None:

        nblobs = len(blobs[:, 0])  # number of blobs
        selected = []

        # get the filter circle values
        yc = filter_circle["y0,x0"][0]
        xc = filter_circle["y0,x0"][1]
        rc = filter_circle["r"]

        # print(f'square rc = {rc**2}')
        for i in range(nblobs):

            # get the blob values
            yi = blobs[i, 0]
            xi = blobs[i, 1]

            # print(blobs[i,:])
            # print((xi-xc)**2+(yi-yc)**2)
            # if the blob center is inside the circle area
            if ((xi - xc) ** 2 + (yi - yc) ** 2) <= rc**2:

                # add the blob
                selected.append(blobs[i, :])

        # update the blobs
        blobs = np.array(selected)

    # convert sigma values to radius.
    blobs[:, 2] = (2**0.5) * blobs[:, 2]

    min_r_lim = (2**0.5) * sigma_lims[0]
    max_r_lim = (2**0.5) * sigma_lims[1]

    if display:

        if "title" not in kwargs.keys():
            kwargs["title"] = "Monchromatic image"

        flup.plot_blobs(
            blobs,
            im2d,
            show_ids=show_ids,
            ids_color=ids_color,
            text_displace=text_displace,
            **kwargs,
        )

        # print some information
        min_r_blobs = min(blobs[:, 2])
        max_r_blobs = max(blobs[:, 2])

        print(f"\n{blobs.shape[0]} colonies were identified")
        print(f"\nsearch radious limits: ({min_r_lim:.3f},{max_r_lim:.3f})")
        print(f"\nextreme blob radious:  ({min_r_blobs:.3f},{max_r_blobs:.3f})\n")

    return blobs


# from IPython.display import display, Image, clear_output
def blob_selector(
    image,
    blobs,
    r_factor=1.1,
    cmap_name=None,
    conversor=255,
    normalize=False,
    columns=5,
    im_width=120,
    grid_gap=10,
    use_contour=True,
    init_check=None,
):
    """
    It displays the blobs images with associated checkboxes.
    Images are displayed in rows of 5. The result of the checkboxes
    is stored in a dictionary for later use.

    Parameters
    ----------
    image: np.array
        2D reference image where the blobs are mapped

    blobs: array (Nx3)
        contains the (y,x) center position and radius of each blob
        for each of N colonies detected in the im2d data.
        estructure for the 'i' element: [yi, xi, ri]

    r_factor: numeric
        factor to multiply the radious of each blob and be able to
        see more context around each one (this help take a more informed
        desition about the pertinence of select the colony)

    cmap_name: string
        if wanna use a specific colormap to display the images

    conversor: numeric
        Value used to convert and or normalize the values of the given image
        in the _array_to_bytes() function. Typically not necessary and is not
        used if normalize = True.

    normalize: boolean or numeric
        if True, each image is normalized to cover full range of colormap.
        If normalize is numeric, this values is used to normalize the image
        between [0, array.max()/normalize]
        In case array.max()/normalize is bigger than 1, the data is renormalized
        to be between [0,1] range.

    columns: int
        number of columns on the displayed widget grid of images

    im_width: int
        width of displayed images. This value is indicated in pixels units.

    grid_gap: int
        gap width between images. This value is indicated in pixels units.

    use_contour: boolean
        if True the colonies automatic identified contour is displayed.

    init_check: dict
        initial values for the check box. Its is used mainly for restart the analysis
        with values used previously (from a stored work).

    Returns
    -------
    selected: dictionary
        It contains the a boolean values for each blob
    """
    nblobs = len(blobs)

    # Crear un diccionario para almacenar el estado de los checkbox
    if init_check != None:
        try:
            selected = {i: init_check[i] for i in range(len(blobs))}
        except:
            print("It was not possible to use the given pre-selected checks")
            selected = {i: True for i in range(len(blobs))}

    # this is the default way
    else:
        selected = {i: True for i in range(len(blobs))}

    # Función para actualizar el diccionario cuando cambia un checkbox
    def on_checkbox_change(change, index):
        selected[index] = change["new"]  # Actualizar el diccionario

    # Crear una lista de filas para agregar imágenes con checkboxes
    rows = []

    for i in range(nblobs):

        # get the blob values
        yi = blobs[i, 0]  # center
        xi = blobs[i, 1]  # center
        ri = blobs[i, 2] * r_factor  # radious, amplified by r_factor

        # use them to compute the blob area limits in the image
        y0 = round_up(yi - ri)  # use floor and add 0.5 to round to the nearest integer.
        y1 = round_up(yi + ri)
        x0 = round_up(xi - ri)
        x1 = round_up(xi + ri)

        # slice the image
        im_i = image[y0 : y1 + 1, x0 : x1 + 1]  # +1 because the slicing syntax

        # Convertir a imagen RGB usando el colormap
        im_rgb = array_to_im(im_i, cmap_name, normalize)

        if use_contour:
            # determine the colony contour
            # Otsu thresholding
            thr_otsu, bin_frame = cv2.threshold(
                im_i, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Find the contours of the binarized image
            contours, _ = cv2.findContours(
                bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            # get the nearest contour to the roi center
            near_contour, centroid = get_nearest_contour(bin_frame, contours)

            # Draw the center nearest contour in red over the image
            cv2.drawContours(
                im_rgb, [near_contour], -1, (255, 0, 0), 1
            )  # Contorno en rojo

        # Convertir de RGB a BGR para OpenCV
        img = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)

        # Convert numpy array to propper format for widget visualization (requires bytes format)
        # Use cv2.imencode to convert to PNG and bytes
        success, encoded_image = cv2.imencode(".png", img)

        if not success:
            raise ValueError("Error at coding the image with OpenCV.")

        img_bytes = encoded_image.tobytes()

        ### create the widget  ####
        img_widget = widgets.Image(
            value=img_bytes,
            format="png",
            layout=widgets.Layout(
                width=f"{im_width}px"
            ),  # , height='100px') #width=100, height=100
        )  # _array_to_bytes(im_i, conversor, cmap_name, normalize)

        # Crear un checkbox para cada imagen
        checkbox = widgets.Checkbox(
            value=selected[i],
            description=f"Blob {i}",
            indent=False,
            layout=widgets.Layout(
                width=f"{im_width}px", align_self="flex-start"
            ),  # same width as the image to be aligned in the widget grid
        )

        # Conectar el evento de cambio del checkbox con la función de actualización
        checkbox.observe(
            lambda change, i=i: on_checkbox_change(change, i), names="value"
        )

        # Agregar título debajo de la imagen si se proporciona
        # title_text = titles[i] if titles else f'Image {i+1}'
        # title = widgets.Label(title_text, layout=widgets.Layout(width=f'{img_width}px'))

        # Combinar la imagen, el título y el checkbox verticalmente
        # row = widgets.VBox([img_widget, title, checkbox], layout=widgets.Layout(align_items=alignment))

        # Combinar imagen y checkbox verticalmente
        row = widgets.VBox(
            [img_widget, checkbox], layout=widgets.Layout(align_items="center")
        )
        rows.append(row)

    ######
    # Crear una cuadrícula para mostrar las imágenes y checkboxes

    ## Calcular el número de columnas basado en el ancho total disponible
    ## columns = int(800 / (im_width + 2*grid_gap))  # Suponiendo un ancho máximo de 800px

    grid = widgets.GridBox(
        children=rows,
        layout=widgets.Layout(
            grid_template_columns=f"repeat({columns}, {im_width + 2*grid_gap}px)",  # Asegura espacio suficiente 'repeat(5, 120px)'
            grid_gap=f"{grid_gap}px",  # Espacio entre elementos
        ),
    )

    # Mostrar la cuadrícula
    display(grid)

    # Retornar el diccionario con los resultados
    return selected


def array_to_im(array, cmap_name="viridis", normalize=False, imformat="RGB"):
    """
    Convert a numpy array (2D) to RGB or RGBA image based on input cmap_name.
    If normalize = True, the image values cover the full colormap range,
    in case is numeric, the max colormap range is set its value (if it is
    bigger than maximum value in array). In case none of the are indicated and
    image values are lower than 1, the colormap range is set between [0,1].

    Parameters
    ----------
    array : numpy.ndarray
        Array de entrada que representa la imagen.

    cmap_name : str
        Nombre del colormap (por ejemplo, 'viridis').

    normalize : bool or numeric
        if True, each image is normalized to cover full range of colormap.
        If normalize is numeric, this values is used to normalize the image
        between [0, array.max()/normalize]
        In case array.max()/normalize is bigger than 1, the data is renormalized
        to be between [0,1] range.

    imformat: string
        'RGB' or 'RGBA'

    Returns
    -------
    im_rgb
        the array converted to RGB or RGBA based on the indicated colormap and inputs.

    """

    # rescale values to [0,1] for cmap proper application

    if normalize is True:
        # make the values to cover the full range [0,1]
        array = array / array.max()

    else:
        try:

            # make the values to cover the range [0,amax/normalize]
            array = array / normalize
        except:
            pass

    # in case normalize is False or the data maximum is still > 1.
    amax = array.max()
    if amax > 1:
        # anyways force the data be inside [0,1] range (covering the full [0,1] by default)
        array = array / amax

    # Apply colormap using matplotlib
    colormap = mpl.colormaps[cmap_name]  # get the colormap

    if imformat == "RGB":
        im_rgb = colormap(array)[:, :, :3]  # Obtener sólo RGB (ignorar alfa)
    else:
        im_rgb = colormap(array)[:, :, :]

    # convert to uint8
    im_rgb = (im_rgb * 255).astype("uint8")  # Escalar de 0-1 a 0-255

    return im_rgb


def _array_to_bytes(array, conversor=255, cmap_name=None, normalize=False):
    """
    Convertir un array numpy a bytes para desplegar con ipywidgets.

    cmap_name: string
        e.g. 'viridis'

    """

    if normalize:
        array = array / np.max(
            array
        )  # Normalizar entre 0 y 1, siendo 1 el máximo de la imagen.

    # if indicated to use a specific colormap
    if cmap_name != None:
        # Aplicar colormap usando matplotlib
        colormap = mpl.colormaps[cmap_name]  # Obtener el colormap

        # cmap renormalization
        if array.max() > 1:
            array = (
                array / conversor
            )  # Normalizar entre 0 y 1, siendo 1 los pixeles iguales a conversor

        array = colormap(array)  # [:, :, :3]  # Obtener sólo RGB (ignorar alfa

    # convert to uint8 array format
    if array.dtype != "uint8" and array.max() <= 1:
        array = conversor * array

    # Convertir a imagen PIL y luego a bytes
    im_pil = pil_image.fromarray(array.astype("uint8"))  #

    buf = BytesIO()
    im_pil.save(buf, format="PNG")

    return buf.getvalue()


def remap_blobs(blobs, rfactor, offset=[0, 0]):
    """
    to map the blobs to images with other coordinates
    (images of another size, rescaled or cropped)
    Basically it perform a linear transformation of its coordinates


    Parameters
    ----------
    blobs: array (Nx3)
        contains the (y,x) center position and radius of each blob
        for each of N colonies.
        Structure for the 'i' element: [yi, xi, ri]

    rfactor: numeric
        Rescaling factor between the im2d data and the image from the source file.

    offset: list with two integers
        list with the [y,x] blobs center coordinates offsets
        from the upper left image corner (integers).

    """
    # init the new array
    rmblobs = np.zeros(blobs.shape)

    for i in range(blobs.shape[0]):

        # perform the linear transformation for each value
        xi = offset[1] + rfactor * blobs[i, 1]
        yi = offset[0] + rfactor * blobs[i, 0]
        ri = rfactor * (2**0.5) * blobs[i, 2]

        # store the values
        rmblobs[i, :] = [yi, xi, ri]

    return rmblobs


def get_colony_center(
    rois, channel, thr, fsmin, fsmax, nsigma=10, overlap=0.1, eb_factor=False, **kwargs
):
    """
    get the center of each colony in a ROI

    Parameters
    ----------
    rois: dict
        dict of ROIs, with the IDs as keys

    channel: str
        channel name to select the data

    thr: numeric
        threshold to use in colony_blobs_id

    fsmin, fsmax: numerics
        relaxation factors to be applied to each colony sigma limits (and use in colony_blobs_id function).
        they should be around 1, with fsmin < fsmax.

    nsigma: integer
        number of intermediate sigma between Smin and Smax to test.
        a low value around 10 should be fine for this function. We are not
        really interested in detect the borders with high precisión.

    overlap: numeric
        maximum allowed overlaping. Value between [0,1].
        For colony_blobs_id

    eb_factor: numeric
        factor applied to the roi radius to assign that value to exclude_border
        parameter of blob_log. This parameter means that colonies inside that
        border distance won't be considered.

    **kwargs:
        for colony_blobs_id

    Return
    ------
    new_centers: dict
        dictionary with the new (y,x) center detected for each ROI.
        dictionary keys are the rois ids.
        They are just returned but not assigned to the object. If you agree
        the obtained values, update them afterwads in each ROI object.

    """

    new_centers = dict()

    for col_id in list(rois.keys()):
        roi = rois[col_id]
        roi_dc = roi.data[channel]
        col_r = roi.blob[2]  # colony blob radius
        r_roi = roi.rroi  # roi radius

        # border exclusion
        if eb_factor:
            exclude_border = eb_factor * r_roi
        else:
            exclude_border = eb_factor

        # minimum sigma should be the closest as posible to the size of the colony to improve the center detection.
        Smin = fsmin * (col_r)
        Smax = fsmax * (col_r)
        Slims = [Smin, Smax]

        # perform the blob detection
        blob_sum_i = colony_blobs_id(
            roi_dc,
            thr,
            sigma_lims=Slims,
            nsigma=nsigma,
            max_over=overlap,
            show_ids=False,
            exclude_border=exclude_border,
            **kwargs,
        )

        # get the number of detected blobs
        blobs_num = blob_sum_i.shape[0]

        # in case more than one blob was detected, store just the y,x coordinates of the nearest blob to ROI center
        if blobs_num > 1:

            distance = np.zeros(blobs_num)

            for i in range(blobs_num):

                distance = (blob_sum_i[i, 1] - r_roi) ** 2

            i_near = np.argmin(distance)

            new_centers[col_id] = blob_sum_i[
                i_near, :2
            ]  # store the y,x coordenates of the first detected blob

        # store the y,x coordenates of the detected blob
        elif blobs_num == 1:
            new_centers[col_id] = blob_sum_i[0, :2]

        # in case no blob was detected
        else:
            print(f"ROI {col_id} without detected blobs\n")

    return new_centers


def get_roi_frame(roi, channel, frame):
    """
    get the roi frame for the indicated frame number.
    If data has just one frame, that frame is used.

    Parameters
    ----------
    roi: flua.Roi
        the roi to get the frame.

    channel: string
        rois data channels name to use.

    frame: int
        frame number to use. Useful to get the last frame value for display
        or confirm the frame number (for example when the data has only one frame
        it give you that data but return frame "None")

    Return
    -------
    roi_frame: numpy.ndarray
        the roi frame.

    frame: int
        the actual frame number.
    """
    # get the propper value accord the data case

    if roi.data[channel].ndim == 3:

        # get the actual frame value for display in texts
        if frame == -1:
            frame = roi.data[channel].shape[2] - 1

        # get the roi frame
        roi_frame = roi.data[channel][:, :, frame]

    elif roi.data[channel].ndim == 2:

        roi_frame = roi.data[channel][:, :]
        frame = None

    else:
        raise ValueError(
            f"Invalid array dimentions in channel {channel} for ROI {roi.id}"
        )

    return (roi_frame, frame)


def get_centroid(countour, invalid_case=[0, 0], integer=True):
    """
    get the contour moments and use them to compute the countour centroid.
    If there is no contour or the moments are zero, the invalid case is returned.
    by default the invalid case is [0,0] (the left-top corner of the image)

    Parameters
    ----------
    countour: numpy.ndarray
        the contour to compute the centroid.
        countour is obtained from cv2.findContours

    invalid_case: list
        the invalid case to return.

    integer: bool
        if True, the centroid values are converted to integer by round_up() function.

    Return
    -------
    cy, cx:  numeric, numeric
        the centroid coordinates
    """
    M = cv2.moments(countour)
    try:
        cy = M["m01"] / M["m00"]
        cx = M["m10"] / M["m00"]

        if integer:
            return round_up(cy), round_up(cx)
        else:
            return cy, cx

    except:
        return invalid_case


def get_nearest_contour(roi, contours, min_distance=None):
    """
    get the nearest contour to the roi center.
    It also could be used in an array instead of a roi.

    Parameters
    ----------
    roi: flua.Roi or array
        the contours and centroids are compared with the center of
        this ROI or array.

    contours: list
        a list of contours. Tipycaly obtained from cv2.findContours

    min_distance: float
        the minimum allowed distance from the colony center to select a coutour.

    Return
    -------
    cy, cx:  int, int
        the centroid coordinates

    """
    # init the outputs
    near_contour = None
    centroid = [None, None]

    #### if there is no contours return None
    if len(contours) == 0:
        return near_contour, centroid

    # get roi center and distance

    if type(roi) == np.ndarray:
        # in case is an array instead of a ROI

        yc = round_up(roi.shape[0] / 2)
        xc = round_up(roi.shape[1] / 2)
        roi_center = [yc, xc]

        if min_distance is None:
            min_distance = max(yc, xc)

    else:
        # if type(roi) == ROI:
        # if isinstance(roi, ROI):
        roi_center = roi.center  # [yc, xc]

        if min_distance is None:
            min_distance = roi.rroi  # use the roi radius as the starting value

    # get the nearest contour to the roi center
    for contour in contours:

        cy, cx = get_centroid(contour)

        # get the distance between the countour centroid and colony center
        center_distance = np.linalg.norm(np.array(roi_center) - np.array([cy, cx]))

        # if this is closer, store its values.
        if center_distance < min_distance:
            # update the min distance
            min_distance = center_distance

            # store this centroid and the countour
            centroid = [cy, cx]
            near_contour = contour

    return near_contour, centroid


def get_biggest_contour(contours, criterion=0, get_cent=True, integer=True):
    """
    get the biggest contour from the input ones based on the perimeter or
    area accord the "criterion" parameter. It also can get and return the
    centroid of the contour.

    Parameters
    ----------
    contours: list
        a list of contours. Tipycaly obtained from cv2.findContours

    criterion: 0 or 1
        criteria to define the biggest contour
        0: perimeter
        1: area

    get_cent: boolean
        if True, compute and return the centroid of the biggest contour

    integer: bool
        if True, the centroid values are converted to integer by round_up() function.


    Return
    -------
    big_contour: array
        the biggest contour in contours

    centroid: list [int, int]
        the centroid coordinates [cy, cx]

    max_size: numeric
        size of the output contour. By default it correspond to perimeter
        but depends on the criterion selected.

    """
    # init the outputs
    big_contour = None
    centroid = [None, None]
    max_size = 0

    #### if there is no contours return None
    if len(contours) == 0:
        return big_contour, centroid, max_size

    # get the biggest contour
    for contour in contours:

        # Perimeter
        if criterion == 0:

            size = cv2.arcLength(contour, True)  # True because is a closed line

        # Area
        elif criterion == 1:

            size = cv2.contourArea(contour)

        # if this is closer, store its values.
        if size > max_size:
            # update the max size
            max_size = size

            # store the current countour
            big_contour = contour

    # compute the centroid if indicated
    if get_cent and big_contour is not None:

        cy, cx = get_centroid(big_contour, integer=integer)
        centroid = [cy, cx]

    return big_contour, centroid, max_size


def calculate_colony_area(
    sdata,
    plot_path=None,
    subtract_radius=300,
    kernel_size=(21, 21),
    threshold_ratio=0.5,
):
    """
    Calculates the dish/colony area based on the provided grayscale image.

    Parameters:
        sdata (numpy.ndarray): Grayscale image (e.g., sdataW["wRG"][:, :, 0]).
        plot_path (str): Directory path where plots are saved.
        subtract_radius (int): Value subtracted from the computed radius.
        kernel_size (tuple): Kernel size for morphological closing.
        threshold_ratio (float): Ratio (0-1) to determine the threshold from the maximum V channel value.

    Returns:
        center (tuple): (cX, cY) center of the detected dish area.
        radius (int): Radius of the computed circle.
        hsv_image (numpy.ndarray): The HSV image computed from the input.
        im_floodfill (numpy.ndarray): The flood-filled binary mask used for contour extraction.
    """
    # Convert the grayscale image to BGR and then to HSV
    bgr_image = cv2.cvtColor(sdata, cv2.COLOR_GRAY2BGR)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    # Get the shape of the image
    IMAGE_Y, IMAGE_X = hsv_image.shape[:2]

    # Define lower and upper limits for the threshold
    lower = 0
    upper = 50

    # Create a binary mask using the V channel
    mask = cv2.inRange(hsv_image[:, :, 2], lower, upper)
    # Invert the mask
    mask = cv2.bitwise_not(mask)

    # Fill the holes in the mask from the center outwards using the fill-basin algorithm
    mask_filled = cv2.floodFill(mask, None, (int(IMAGE_X / 2), int(IMAGE_Y / 2)), 255)[
        1
    ]
    # Perform a closing operation to fill the holes
    mask_filled = cv2.morphologyEx(
        mask_filled,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )

    # Find external contours on the filled image.
    contours, _ = cv2.findContours(
        mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError(
            "No contours found in the image. Check your threshold and input image."
        )

    # Choose the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute the moments of the largest contour to get the centroid.
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        raise ValueError(
            "Zero division error while computing moments; contour area is zero."
        )
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Estimate a circle from the contour area:
    # We compute an equivalent radius from the contour area and subtract a fixed value.
    radius = int(np.sqrt(M["m00"] / np.pi)) - subtract_radius
    radius = max(radius, 0)  # Ensure non-negative radius

    # Visualize the final result with the computed circle overlaid
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(hsv_image)
    circle = plt.Circle((cX, cY), radius, color="r", fill=False, linewidth=2)
    ax.add_artist(circle)
    ax.set_title("Detected Colony Area")
    plt.axis("off")
    # Save the image and close the figure
    if plot_path:
        plt.savefig(os.path.join(plot_path, "colony_area.png"), bbox_inches="tight")
    plt.close()

    return (cX, cY), radius, hsv_image, mask_filled


def data_thr_contour(
    data,
    channel,
    frames=-1,
    bin_value=1,
    threshold=False,
    tcr=None,
    thr_step=1,
    thr_limits=None,
    r_tol=0,
    print_info=False,
    fig_ncol=2,
    lw=1,
    out_circle=True,
    show_fig=True,
    ofname=None,
    **kwargs,
):
    """
    It uses OTSU Threshold as to define the channels data binarization.
    Then use findContours to catch the biggest contour in the binarized image
    Optionally you can compute the Enclosing Circle of the contour.
    It displays the channels and binarized image with the contour and centroid.
    IF out_circle = True, also display the enclosing circle and its center.

    * Could be improved to accept other enclosing figures like rectangle
      e.g:  x,y,w,h = cv2.boundingRect(cnt),  (x,y) top-left coordinate of the rectangle, (w,h) its width and height

    Parameters
    ----------
    data: dict
        the dictionary with the images channels data in each element
        data[channel] = [y,x,frame]

    channel: string
        rois data channels name to use.

    frames: list
        list with frame numbers to use.

    bin_value: int
        binary value to assign to the pixels that surpass the threshold.

    threshold: False, numeric or list/np.array
        if a numeric values is indicated, this values is used to binaruze all the images
        if a list/np.array is indicated, use the asociated threshold value to each frame
        By default is False and OTSU threshold is automatically determined for each frame.

    tcr: numeric
        target enclosing circle radius value.
        if given, the algorithm search the threshold to get the enclosing circle
        with the nearest radius to this value.

    thr_step: numeric
        in case of tcr was indicated this is the threshold step used to seach
        the threshold that gives the enclosing circle nearest to the input tcr value.

    thr_limits: list
        threshold limits [min, max]. By default it uses the [0,255] for integer
        data types and [0,1] for float data types.

    r_tol: numeric (positive value)
        In case tcr was indicated, this is the acceptable tolerance for
        radius difference. The search algorithm stops at reach this difference.

    print_info: boolean
        if True, the algorithm info is displayed in each cicle.
        It is useful for debugging.

    fig_ncol: int
        number of columns in the figure.

    ofname: string
        if given, the figure is stored under that name. More options
        with **kwargs  (see flua.save_fig())

    out_circle: boolean
        if True, the Enclosing Circle (the circumcircle) of the contour is
        determined using the function cv.minEnclosingCircle().

    show_fig: boolean
        if True, the image figure is diaplayed
    lw: int
        plots line width. It also determines the center/centroid dot size.

    Returns
    -------
    results: dict
        It contains:

        "frames": np.array
            frame number actually used

        "centroids": np.array
            the centroid coordinates for each contour frame
            centroids[0,:] = [yc0,xc0]
            centroids[:,0] = [yc0,yc1,..,ycn]

        "centers": np.array
            the center coordinates the eclosed circle asociated to the contour
            of each frame
            center[0,:] = [yc0,xc0]
            center[:,0] = [yc0,yc1,..,ycn]
            only present if out_circle = True

        "r": np.array
            enclosed circle radius values
            only present if out_circle = True

        "thresholds": np.array
            the Otsu threshold value for each frame

    """
    # verify frames format (list or array)
    if type(frames) not in (list, np.ndarray):
        frames = [frames]

    # get the indicated channel
    datac = data[channel]
    text_off_y = int(0.05 * datac.shape[0])  # centroid text position offset in y-axis

    # init the output dictionary
    results = {
        "frames": [],
        "centroids": [],
        "thresholds": [],
    }

    if out_circle:
        results["centers"] = []
        results["r"] = []

    # big_contours = {}
    # binims = {}

    ## init the figure
    if show_fig:
        nframes = len(frames)
        nrows = nframes  # //fig_ncol +1

        fig, axs = plt.subplots(
            nrows, fig_ncol, figsize=(4 * fig_ncol, 4 * nrows), layout="constrained"
        )

        if nrows > 1 and fig_ncol > 1:
            axs = axs.flatten()

        if nrows == 1 and fig_ncol == 1:
            axs = [axs]

    idx = 0  # index for the figure (let it outside if, for update simplicity)

    # get the roi channel data for the indicated frame
    for i in frames:

        # get the channel frame data
        dataci = datac[:, :, i]

        if threshold == "custom":
            # Custom thresholding using the calculate colony area function
            center, radius, hsv_image, im_floodfill = calculate_colony_area(dataci, **kwargs)
            # Add the resulting values to the results dictionary
            results["frames"].append(i)
            results["centroids"].append(center)
            results["r"].append(radius)
            results["centers"].append(center)
            results["thresholds"].append("custom_function")
            
            continue

        # threshold to be used or starting threshold in case tcr was indicated
        elif threshold is False:
            # Otsu thresholding in case no threshold was indicated
            bin_thr, bin_frame = cv2.threshold(
                dataci, 0, bin_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        # if defined threshold values were indicated
        elif type(threshold) == list or type(threshold) == np.ndarray:
            # defined threshold for each frame
            bin_thr, bin_frame = cv2.threshold(
                dataci, threshold[i], bin_value, cv2.THRESH_BINARY
            )

        # if use the same threshold value for all the frames
        else:
            # Static binary thresholding
            bin_thr, bin_frame = cv2.threshold(
                dataci, threshold, bin_value, cv2.THRESH_BINARY
            )

        try:
            # in case no target circle radius was indicated
            if tcr is None:

                # Find the contours of the binarized image
                contours, hierarchy = cv2.findContours(
                    bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                # get the biggest contour
                contour, centroid, size = get_biggest_contour(
                    contours, criterion=0, get_cent=True, integer=False
                )

                # get the enclosing circle
                if out_circle:

                    (cx, cy), cr = cv2.minEnclosingCircle(contour)  # cr = radius
                    # cy = round_up(y)
                    # cx = round_up(x)
                    # radius = round_up(cr)

                    # add the values to out dict
                    results["centers"].append([cy, cx])
                    results["r"].append(cr)

                # add the values to out dict
                results["centroids"].append(centroid)  # [yc,xc]
                results["thresholds"].append(bin_thr)
                results["frames"].append(i)

                # binims[i] = bin_frame              # probably this dictionary is not necessary

            # in case target circle radius was indicated
            else:

                # compute the threshold limits if were not indicated
                if thr_limits is None:

                    # in case dtype is float
                    if np.issubdtype(dataci.dtype, np.floating):
                        thr_limits = [0, 1]

                    # in case dtype is integer
                    elif np.issubdtype(dataci.dtype, np.integer):
                        thr_limits = [0, 255]

                    else:

                        print("Image data is a not supported data type")
                        return ()

                # int the minimun difference variable
                min_diff = float("inf")

                # starting threshold
                if threshold is False:
                    # it uses Otsu thresholding in case non threshold was indicated
                    bin_thr, bin_frame = cv2.threshold(
                        dataci, 0, bin_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )

                # keep a record of applied thresholds to avoid repetitions and loops
                thr_history = []

                # perform the threshold search that suits the target radius
                while True:

                    thr_history.append(bin_thr)

                    # Find the contours of the binarized image
                    contours, hierarchy = cv2.findContours(
                        bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )

                    if not contours:
                        if print_info:
                            print("Not contours found. Exit")
                        break

                    # get the biggest contour
                    contour, centroid, size = get_biggest_contour(
                        contours, criterion=0, get_cent=True, integer=False
                    )

                    # get the Small Enclosing Circle of that contour
                    (cx, cy), cr = cv2.minEnclosingCircle(contour)

                    # compute the difference of radius
                    rdiff = tcr - cr

                    if print_info:
                        print(f"frame {i}, thr = {bin_thr}, cr = {cr}, rdiff = {rdiff}")

                    # stop the search if radius difference increase (continue searching even if equal, because it could be reduced in next step)
                    if abs(rdiff) > abs(min_diff):
                        if print_info:
                            print("Radius diference increase. Exit")
                        break

                    # update min difference
                    min_diff = rdiff

                    # if reach the target circle radius under the allowed tolerance error
                    if abs(rdiff) <= r_tol:
                        if print_info:
                            print("Radius tolerance reach. Exit")
                        break

                    else:
                        # if the target is bigger than the current circle radius
                        if rdiff > 0:
                            # reduce the threshold to increase cr
                            thr = bin_thr - thr_step

                        # if the target is bigger than the current circle radius
                        elif rdiff < 0:
                            # increase the threshold to reduce cr
                            thr = bin_thr + thr_step

                    # stop if thr is outside threshold limits
                    if thr < thr_limits[0] or thr > thr_limits[1]:
                        if print_info:
                            print("Threshold out of limits. Exit")
                        break

                    # stop if thr reach a bucle
                    if thr in thr_history:
                        # if len(thr_history) > 2 and thr_history[-1] == thr_history[-3]:
                        if print_info:
                            print("Threshold bucle. Exit")
                        break

                    # apply the new threshold
                    bin_thr, bin_frame = cv2.threshold(
                        dataci, thr, bin_value, cv2.THRESH_BINARY
                    )

                # add the values to out dict
                results["centers"].append([cy, cx])
                results["r"].append(cr)
                results["centroids"].append(centroid)  # [yc,xc]
                results["thresholds"].append(bin_thr)
                results["frames"].append(i)

            # plot the images
            if show_fig:
                axs[idx].imshow(bin_frame)  # binarized
                axs[idx + 1].imshow(dataci)  # input image channel data

                for j in [0, 1]:
                    if contour is not None:
                        axs[idx + j].plot(
                            contour[:, 0, 0],
                            contour[:, 0, 1],
                            "r",
                            linewidth=lw,
                            label="contour",
                        )

                    # add the centroind
                    if centroid is not None:
                        axs[idx + j].plot(
                            centroid[1],
                            centroid[0],
                            "or",
                            markersize=lw + 1,
                            linewidth=lw,
                            label=f"centroid [{centroid[0]:.1f},{centroid[1]:.1f}]",
                        )

                    axs[idx + j].annotate(
                        f"[{centroid[0]:.1f},{centroid[1]:.1f}]",
                        (centroid[1], centroid[0] - text_off_y),
                        ha="center",
                        va="bottom",
                        color="red",
                    )

                    # add the enclosing circle
                    if out_circle:
                        # create the enclosing circle
                        enc_circle = Circle(
                            (cx, cy),
                            cr,
                            color="k",
                            fill=False,
                            lw=lw,
                            label=f"enclosing circle (r = {cr})",
                        )  # [{cy},{cx}] f'r = {r_px} px')
                        axs[idx + j].add_artist(enc_circle)

                        # add the circle center
                        axs[idx + j].plot(
                            cx,
                            cy,
                            "xk",
                            markersize=lw + 1,
                            linewidth=lw,
                            label=f"center [{cy:.1f},{cx:.1f}]",
                        )
                        axs[idx + j].annotate(
                            f"[{cy:.1f},{cx:.1f}]",
                            (cx, cy + text_off_y),
                            ha="center",
                            va="top",
                            color="k",
                        )
                    # apply some format
                    axs[idx + j].axis(
                        "image"
                    )  # Ajusta los ejes para que coincidan con los de la imagen
                    axs[idx + j].set_xticks([])
                    axs[idx + j].set_yticks([])  # Ocultar los ticks de los ejes

                axs[idx].legend(loc="lower left")

                axs[idx].set_title(f"thr = {bin_thr}, frame {i}, binarized")
                axs[idx + 1].set_title(f"thr = {bin_thr}, frame {i}, {channel}")

        except:
            print(f"\nframe {i} fail to be processed")
            # add the values to out dicts
            centroids[i] = [None, None]
            centers[i] = [None, None]
            thresholds[i] = None

            if show_fig:
                # make this axs off
                axs[idx].axis("off")

        # update the axis index
        idx += 2

    if show_fig:
        # make off the remaining not used axes
        while idx < len(axs):
            axs[idx].axis("off")
            idx += 1
        plt.show()

    # save the figure file if indicated
    if ofname is not None:
        ofname += f""
        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        fig_kwargs.update(kwargs)  # join with the input functions kwargs
        save_figure(fig, ofname, **fig_kwargs)

    # convert result elements in np.arrays
    for key in results.keys():
        results[key] = np.asarray(results[key])

    return results


def get_circle_data(data, center, r, mode="<=", boolean=False):
    """
    To get the the data of (N,M,..) arrays confined inside or outside a circular
    area defined in the N,M dimentions.
    The selected area is controled by 'mode' parameter.
    If boolean parameter is True, the output is a boolean mask of the area.

    Parameters
    ----------
    data: np.array
        multidimentional array with the data. It has to be al least ndim = 2.

    center: list or array
        it contains the center of the circle
        i.e. [cy,cx]

    r: numeric
        radius value

    mode: str, optional
        it defines whish section is returned (the internal, the external, etc)
        accord the operators '<=', '>=', '<', '>', '='. (five options)

    boolean: bool
        if True, the output is a boolean array.

    Returns
    -------
    cdata: np.array
        array of the same dimentions and shape as the input data but
        keeping just the values inside or outside the given circle.
        The rest of the data is set to zero.
        In case boolean parameter is True, the output is a boolean array
        with the indicated area True and the rest False.

    """
    # check the data format
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")

    # numpy operation relation table
    operations = {
        "<=": np.less_equal,
        "<": np.less,
        ">=": np.greater_equal,
        ">": np.greater,
        "=": np.equal,
    }

    # Verificar que el modo sea válido
    if mode not in operations:
        raise ValueError(
            f"Invalid mode: {mode}. Valid options are '<=', '>=', '<', '>', '='."
        )

    # get the data shape
    h, w = data.shape[:2]

    # get the center values
    cy, cx = center

    # Create a grid of coordinates
    y, x = np.ogrid[:h, :w]

    # Compute the distance to the center
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2

    # Get the mask related to circle applying the selected operator
    mask = operations[mode](dist_sq, r**2)
    # mask = (x - cx)**2 + (y - cy)**2 <= r**2

    # if selected, convert the data to a boolean True array
    if boolean:
        data = np.ones_like(data).astype("bool")

    # init the array of zeros
    cdata = np.zeros_like(data)

    # Apply the mask to the data
    cdata[mask, ...] = data[mask, ...]

    return cdata


def get_circle_mask(center, r, dims=None, ref_array=None, mode="<="):
    """
    To get a circular boolean mask given a input center and radius.
    The output array size is defined by explicit dimentions (dims parameter) or
    using a reference array (ref_array)
    If iternal or external circle area is considered as True is defined by
    'mode' parameter.

    Parameters
    ----------
    center: list or array
        it contains the center of the circle
        i.e. [cy,cx]

    r: numeric
        radius value

    dims: list or tuple, optional
        the dimentions of the output array dims = [10,10]

    ref_array: np.array, optional
        multidimentional array used as size reference to create the output array.
        It has to be al least ndim = 2.

    mode: str, optional
        it defines whish section is returned (the internal, the external, etc)
        accord the operators '<=', '>=', '<', '>', '='. (five options)

    Returns
    -------
    mask: np.array
        boolean array mask of the indicated dimentions with True values inside
        or outside the indicated circle (accord the selected mode).

    """
    # Verify the parameters
    if dims is None and ref_array is None:
        raise ValueError("You have to indicate 'dims' or 'ref_array'. Exit.")

    # create the base mask
    if dims is not None:
        mask = np.ones(dims).astype("bool")

    else:
        if not isinstance(ref_array, np.ndarray):
            raise ValueError("ref_array must be a numpy array.")

        mask = np.ones_like(ref_array).astype("bool")

    # get the center values
    mask = get_circle_data(mask, center, r, mode)

    return mask

def background_mask(
    data,
    channel,
    method="threshold",          # "threshold" or "mog2"
    threshold=30,
    statistic="mean",
    frame_idxs=None,
    start=0,
    end=None,
    bin_width=None,
    display_hist=False,
    stat_frames=...,
    dif_frames=...,
    # MOG2 hyper-parameters (only used if method=="mog2")
    mog_history=500,
    mog_varThreshold=16,
    mog_detectShadows=False,
):
    """
    Identify background pixels via either:
      - a global threshold on mean/median deviation ("threshold")
      - or OpenCV's adaptive MOG2 background subtractor ("mog2")
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask where True = background.
    mean_diff : np.ndarray or None
        Per-pixel mean difference (only for threshold method).
    """
    # extract the 3D image stack
    cdata = data[channel]
    if cdata.ndim != 3:
        raise ValueError("Input data must be a 3D array (H, W, T).")

    # determine which frames to use
    if frame_idxs is None:
        if end is None:
            end = cdata.shape[2]
        frame_idxs = np.arange(start, end)
    frame_idxs = np.asarray(frame_idxs, dtype=int)

    # override selection if stat_frames / dif_frames are given
    if stat_frames is not ...:
        stat_frames = np.atleast_1d(stat_frames).astype(int)
    else:
        stat_frames = frame_idxs
    if dif_frames is not ...:
        dif_frames = np.atleast_1d(dif_frames).astype(int)
    else:
        dif_frames = frame_idxs

    # --- Method 1: threshold on temporal statistic ---
    if method.lower() == "threshold":
        # compute background model
        if statistic == "mean":
            background_model = np.mean(cdata[:, :, stat_frames], axis=2)
        elif statistic == "median":
            background_model = np.median(cdata[:, :, stat_frames], axis=2)
        else:
            raise ValueError("`statistic` must be 'mean' or 'median'.")

        # compute abs difference and mean difference
        diff = np.abs(cdata[:, :, dif_frames] - background_model[..., None])
        mean_diff = np.mean(diff, axis=2)

        # initial mask
        mask = mean_diff < threshold

        # morphological cleanup (remove small specks, fill small holes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_open = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = mask_clean.astype(bool)

        # optional histogram
        if display_hist:
            if bin_width is None:
                bins = "auto"
            else:
                bins = np.arange(mean_diff.min(), mean_diff.max() + bin_width, bin_width)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(mean_diff.ravel(), bins=bins)
            ax.axvline(threshold, color="k", linestyle="--", label=f"thr = {threshold}")
            ax.set(title="Pixel deviation from background model", xlabel="mean abs diff", ylabel="count")
            ax.legend()
        return mask, mean_diff

    elif method.lower() == "mog2":

        # Create the background subtractor
        fgbg = cv2.createBackgroundSubtractorKNN(
            history=mog_history,
            detectShadows=mog_detectShadows
        )
        # Feed frames in temporal order
        for idx in frame_idxs:
            frame = cdata[:, :, idx]
            # ensure 8-bit input
            if frame.dtype != np.uint8:
                frame_uint8 = frame.astype(np.uint8)
            else:
                frame_uint8 = frame
            _ = fgbg.apply(frame_uint8)

        # Apply to the last frame to get the final fg mask
        fgmask = fgbg.apply(frame_uint8)

        # First, clean it up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #mask_clean = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel,
        #                            iterations=5)
        #mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel,
        #                            iterations=5)
        # binary mask: True = foreground, False = background
        #mask = (mask_clean == 255)
        mask = (fgmask == 255)

        # -----------------------------
        # Now detect circle‐like blobs and force them to foreground
        # -----------------------------

        # We need a uint8 mask to draw on:
        mask_uint8 = (mask.astype(np.uint8)) * 255

        # Hough circle detection on the last frame
        circles = cv2.HoughCircles(
            frame_uint8,
            cv2.HOUGH_GRADIENT,
            dp=1.25,
            minDist=25,           # min distance between centers
            param1=50,            # upper threshold for Canny edge detector
            param2=50,            # threshold for center detection
            minRadius=25,
            maxRadius=125
        )

        if circles is not None:
            # round and cast to ints: shape (N, 3) => x, y, r
            circles = np.round(circles[0, :]).astype(np.int32)
            for (x, y, r) in circles:
                # draw filled circle (255) onto mask
                cv2.circle(mask_uint8, (x, y), r, 255, thickness=-1)

        # Optionally apply closing/opening to clean small holes/noise:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN,  kernel, iterations=3)
        
        # Convert back to boolean mask
        mask = (mask_uint8 == 255)
        # mask = mask.astype(bool)

        return mask, circles

    else:
        raise ValueError("`method` must be 'threshold' or 'mog2'.")


def background_mask_OLD_VERSION(
    data,
    channel,
    threshold=30,
    statistic="mean",
    frame_idxs=None,
    start=0,
    end=None,
    bin_width=None,
    display_hist=False,
    stat_frames=...,
    dif_frames=...,
):
    """
    It identify the pixels that correspond to the background based on
    the dynamics of the signal of each one.
    It is based on the difference of each pixel value agains its temporal
    mean or median value (it could be computed using the whole data or using
    a subset of times)
    Pixels with a low difference related to the statistic (under the threshold)
    are considered as background

    Parameters
    ----------
    data: dictionary
        timelapse image data dictionary with each channel is a np.ndarray element
        e.g. data[channel](N, M, K)

    channel: str
        channel name

    threshold: float
        pixels which its difference is under this value are considered as background

    statistic: str
        statistic used as model of the average behaviour of each pixel.

    frame_idx: np.array
        1d numpy array with the index of images to be used to compute the average
        background behaviour (to compute the mean or median)

    start: int
        Just used in case frame_idx is not given.
        Starting index to be used for the background behaviour approximation.

    end: int
        Just used in case frame_idx is not given.
        End index to be used for the background behaviour approximation.

    bin_width: numeric or None, optional
        None uses the default matplotlib bins width determination
        if numeric, it defines uniformly spaced bins of this value.

    diff_frames: list or array of indexs, optional
        If given just that frames are used to compute the difference over
        the data statistic.

    Returns
    -------
    mask : np.ndarray
        binary mask where background pixels are equal to True (or 1)
    """
    t0 = ttime()
    # get the channels data
    cdata = data[channel]

    # verify the input data
    if cdata.ndim != 3:
        raise ValueError("Input data must be a 3D array (N, M, K).")

    # get the frames in case they were not directly input
    if frame_idxs == None:

        if end == None:
            end = cdata.shape[2]

        frame_idxs = np.arange(start, end)

    # verify the frames format used to compute the statistic and the difference
    if stat_frames != ...:
        if not isinstance(stat_frames, (list, np.ndarray)):
            stat_frames = np.asarray([stat_frames])

    if dif_frames != ...:
        if not isinstance(dif_frames, (list, np.ndarray)):
            dif_frames = np.asarray([dif_frames])

    # get the chosen statistic of the pixels as reference
    if statistic == "mean":
        # temporal mean
        background_model = np.mean(cdata[:, :, stat_frames], axis=-1)

    elif statistic == "median":
        # temporal median
        background_model = np.median(cdata[:, :, stat_frames], axis=-1)

    else:
        raise ValueError("Statistic choise have to be 'mean' or 'median'. Aborting...")
    print(f"model computed {ttime()-t0}")
    t0 = ttime()
    # absoute difference between each pixel and the statistic value
    diff = np.abs(cdata[:, :, dif_frames] - background_model[..., None])
    print(f"diff computed {ttime()-t0}")
    t0 = ttime()
    # Mean temporal difference.
    mean_diff = np.mean(diff, axis=-1)
    print(f"mean diff computed {ttime()-t0}")
    t0 = ttime()
    # Pixels with a low difference related to the statistic (under the threshold) are considered as background
    # Create a mask with the pixels which difference is under the threshold
    mask = mean_diff < threshold
    print(f"mask computed {ttime()-t0}")
    t0 = ttime()

    if bin_width is None:
        # default matplotlib width determination ('automatic')
        bins = "auto"  # rcParams["hist.bins"]# equal to 10
    else:
        # defined width
        bins = np.arange(mean_diff.min(), mean_diff.max() + bin_width, bin_width)

    print(f"bins computed {ttime()-t0}")
    t0 = ttime()

    # display a histogram of the result to make an idea of the threshold
    if display_hist == True:
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        ax.hist(mean_diff.flatten(), bins)
        ax.axvline(
            threshold,
            color="k",
            linestyle="dashed",
            linewidth=1,
            label=f"Threshold = {threshold}",
        )
        ax.set_ylabel("count")
        ax.set_xlabel("pixel difference")
        ax.set_title(f"Pixel difference from {statistic}")
        plt.legend()

        print(f"plot done {ttime()-t0}")

    elif display_hist == False:
        print(f"no plot generated {ttime()-t0}")

    # flua.display_im_mask(image, mask, alpha=0.5)

    return (mask, mean_diff)


def combine_masks(mask1, mask2, logic_op=np.logical_or):
    """
    Combine two same shape mask based on input logic operation

    Parameters:
    ----------
    mask1: np.array (bool or int)
        fist binary mask

    mask2: np.array (bool or int)
        second binary mask

    logic_op: numpy logic operator
        any numpy logic operator like
        "np.logical_and", "np.logical_xor", etc

    Returns:
    -------
    combined_mask : np.array (bool or int)
        combined mask accord the given logic operator

    """

    if mask1.shape != mask2.shape:
        raise ValueError("Maks have to be same shape. Exit.")

    # Aplicar la operación lógica AND
    combined_mask = logic_op(mask1, mask2)

    return combined_mask


def refine_mask(
    mask,
    ksize=5,
    cv2_close=True,
    cv2_open=True,
    erode_iter=None,
    dilate_iter=None,
    min_size=100,
    connectivity=8,
    gksize=None,
):
    """
    Refine the input mask by applying different cv2 morphological operations.
    Specifically,
    cv2.MORPH_CLOSE is applyied to eliminate small holes inside elements
    cv2.MORPH_OPEN is applyied to eliminate the "salt pepper noise' outside the elements
    cv2.erode is applyied to expand the borders of the elements (bakcground is erode, then holes growth).
    cv2.connectedComponentsWithStats to select the connected components by a minimum size
    cv2.GaussianBlur to smmoth the borders (None applyed by default)

    Morphological operations in OpenCV require the mask to be of type uint8
    and the values ​​to be 0 and 255 (not just True or False). Then, if boolean
    mask is input, it will be transformed to that format and reversed to boolean
    before returned.

    more info of the transformations:
    https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

    Parameters:
    -----------
    mask : np.array
        input mask (it could be binary or uint8)

    ksize : int, optional
        Morphological operations kernell size.

    cv2_close: bool, optional
        If True, cv2.MORPH_CLOSE is applyied to join near regions

    cv2_open: bool, optional
        If True, cv2.MORPH_OPEN is applyied to reduce the noise in elements borders

    erode_iter: int, optional
        If given, cv2.erode is applyied to expand the elements borders
        the iterations indicated in this parameters (typically 1 to 3)
        (the background mask is erode, then the elements/holes growth)

    min_size : int, optional
        minimal elements size. If they are under this value, will be eliminated
        from the mask.

    connectivity: numeric, optional
        cv2.connectedComponentsWithStats connectivity parameter value which
        could be 4 (square connected) or 8 (diamond connected).
        (4 doesn´t considerer diagonal connected components as 8 do)

    gksize : int, optional
        kernel size for gaussian blur.
        If gksize = None, filter is not applicated.

    Return:
    ---------
    refined_mask : np.array
        refined mask in the same data type format as it was input but
        refined accord the indicated morphological operations.

    """
    # get and store the data type of input mask
    mdtype = mask.dtype

    # Convert mask to uint8 before morphological operations (in case it is not uint8 data type)
    if mdtype == bool:
        mask = mask.astype(np.uint8) * 255  # Converts True/False to 0/255

    elif mdtype != np.uint8:
        mask = mask.astype(np.uint8)

        if mask.max() != 255 and mask.max() <= 1:
            mask = mask * 255

    # create the kernell to be used in the operations
    kernel = np.ones((ksize, ksize), np.uint8)

    # Apply morphologyc close to eliminate small holes inside elements
    if cv2_close:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply morphologyc close to eliminate the noise outside the elements
    if cv2_open:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply erotion to expand the elements area (as a relaxation of the borders)
    if erode_iter != None and dilate_iter != None and erode_iter == dilate_iter:
        # Loop over the number of iterations
        for _ in range(erode_iter):
            # Apply erosion to the mask
            mask = cv2.erode(mask, kernel, iterations=1)
            # Apply dilation to the mask
            mask = cv2.dilate(mask, kernel, iterations=1)
        #mask = cv2.erode(mask, kernel, iterations=erode_iter)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )

    # Start the refined mask and eliminate the elements under the required minimal size.
    refined_mask = np.zeros_like(mask)

    for i in range(1, num_labels):  # fisrt label (background) is ignored
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            # the region is conserved
            refined_mask[labels == i] = 255

    # smooth the borders with gaussian filter if indicated
    if gksize != None:
        refined_mask = cv2.GaussianBlur(refined_mask, (gksize, gksize), 0)

    # if input mask was boolean data type, reverse it to that format
    if mdtype == bool:
        # each value over 127 is 'True' and 'False' if under that value.
        refined_mask = refined_mask > 127
        refined_mask.astype(bool)

    return refined_mask


def extract_masked_region(
    data, mask1, mask2=None, fill_val_1=0, fill_val_2=255, crop=True
):
    """
    Extract the values of input data array [N, M, K] indicated as True in mask [N, M].
    The value for mask1's False positions is defined by fill_val_1.
    Additionally, an optional mask2 [N, M] can be given, which is applied after mask1.
    False values in mask2 are filled with fill_val_2.

    If 'crop' is True, the returned array is reduced to the minimal rectangular region
    that includes the True values in mask1.

    Parameters:
    -----------
    data : np.array (N, M, K)
        Data array to be masked

    mask1 : np.array (N, M)
        Boolean mask of the region of interest (True values are kept and False
        positions are assigned fill_val_1)

    mask2: np.array (N, M), optional
        Optional second boolean mask applied after mask1.
        (True values are kept and False positions are assigned fill_val_2)

    fill_val_1 : int or float, optional
        Value assigned to mask1 False positions

    fill_val_2 : int or float, optional
        Value assigned to mask2 False positions

    crop: boolean, optional
        If True, the returned data is cropped to the minimal rectangular region
        that includes the True values in mask1.

    Returns:
    --------
    modified_data : np.array (H, W, K) or (N, M, K)
        Data after applying the mask processing. Dimensions depend on the 'crop' flag.

    cropped_masks : tuple
        Contains the cropped mask(s) reduced to the minimal rectangular region.
        If mask2 is provided, it returns a tuple (cropped_mask, cropped_mask2);
        otherwise, it returns (cropped_mask,).

    false_mask: np.array (H, W) or (N, M)
        The mask indicating the False positions after applying mask1 (and mask2 if given).
        In this mask, False positions are indicated as True.

    bbox : tuple
        Coordinates of the crop region (y_min, y_max, x_min, x_max).
    """
    # Get the coordinates of True values in mask1
    y_indices, x_indices = np.where(mask1)

    if len(y_indices) == 0 or len(x_indices) == 0:
        raise ValueError("Mask has no True values. Exiting.")

    if crop:
        # Find the minimal rectangular region of True values
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
    else:
        # Use the full data dimensions
        y_min, y_max = 0, data.shape[0] - 1
        x_min, x_max = 0, data.shape[1] - 1

    # Get the rectangular region from data and mask1
    cropped_data = data[y_min : y_max + 1, x_min : x_max + 1, :]  # Keeps K dimension
    cropped_mask = mask1[y_min : y_max + 1, x_min : x_max + 1]

    # Create a copy of the data to safely modify its values
    modified_data = np.copy(cropped_data)

    # Apply the fill value for positions where mask1 is False
    modified_data[~cropped_mask] = fill_val_1

    if mask2 is not None:
        # Crop the second mask similarly
        cropped_mask2 = mask2[y_min : y_max + 1, x_min : x_max + 1]
        modified_data[~cropped_mask2] = fill_val_2

        # Get the combined false mask
        try:
            false_mask = combine_masks(
                ~cropped_mask, ~cropped_mask2, logic_op=np.logical_and
            )
        except Exception:
            print(
                "Warning: The masks could not be combined. False mask will be determined "
                "by value (it might include positions that coincidentally have fill_val_1)."
            )
            false_mask = modified_data == fill_val_1

        cropped_masks = (cropped_mask, cropped_mask2)
    else:
        # When only mask1 is provided
        false_mask = ~cropped_mask
        cropped_masks = (cropped_mask,)

    return modified_data, cropped_masks, false_mask, (y_min, y_max, x_min, x_max)


def replace_regions(im1, im2, mask, dkernel=1):
    """
    Replace the regions of im1 indicated as True in mask by the elements of
    im2 in that positions.
    The replace is poderated by a scale factor determined to reduce
    the difference with the surrounding are.
    The considered surrounding area is defined by the dkernel (increase as
    kernel value is large)

    Parameters:
    ----------
    im1 : np.array (N,M)
        Mocromatic image data array to be modified

    im2 : np.array (N,M)
        Mocromatic image data array to get the replace data

    binary_mask : np.array
        Binary mask were replace objects or areas are indicated as True.

    Return:
    --------
    im1 : np.array
        Modified im1 data where the indicated regions were smoothly replaced by
        the corresponding values of im2

    """

    # Verify they are the same size
    if im1.shape != im2.shape:
        raise ValueError("Images have to be same size")

    # maske sure they are unit8 format
    if im1.dtype != np.uint8:
        im1 = change_dtype(im1, np.uint8, print_type=False, forze=True)
        # im1 = im1.astype(np.uint8)

    if im2.dtype != np.uint8:
        im2 = change_dtype(im2, np.uint8, print_type=False, forze=True)
        # im2 = im2.astype(np.uint8)

    expanded_mask = cv2.dilate(
        mask, np.ones((dkernel, dkernel), np.uint8), iterations=1
    )

    # Find the countour in the mask
    contours, _ = cv2.findContours(
        expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    # create a copy of the image before modify it (es importante para no modificar la original!! ya me pasó)
    result = im1.copy()

    # peform the replace of each elements identified as contour
    replaced_obj = np.zeros_like(im1)

    print(f"number of elements: {len(contours)}")

    for ctr in contours:
        # create a mask for the current contour
        ctr_mask = np.zeros_like(expanded_mask, dtype=np.uint8)
        cv2.drawContours(ctr_mask, [ctr], -1, 255, thickness=1)

        # Get the coordintes of the contour
        ctr_pixels = np.where(ctr_mask == 255)

        if len(ctr_pixels[0]) == 0:
            print("Warning: contour has no pixels. Skipping")
            continue

        # get the countour value of contour pixels in both images
        ctr_vals1 = im1[ctr_pixels]
        ctr_vals2 = im2[ctr_pixels]

        # compute the scale factor as the relation in their contour pixels value difference
        diff = np.mean(ctr_vals1) - np.mean(ctr_vals2)
        if np.mean(ctr_vals2) != 0:
            scale_factor = np.mean(ctr_vals1) / np.mean(ctr_vals2)
        else:
            scale_factor = 1  # to avoid zero divition
        print(f"Scale Factor = {scale_factor}")

        # Crear una máscara para el objeto dentro del contorno
        # create a mask for the object inside the contour
        obj_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(obj_mask, [ctr], -1, 255, thickness=cv2.FILLED)

        # Get the pixels inside the contour
        obj_pixels = np.where(obj_mask == 255)

        if len(obj_pixels[0]) == 0:
            print("Warning: object has no pixels. Skipping")
            continue

        # Apply the scale factor and replace the region in im1
        print(f"largo: {len(obj_pixels)}")
        adjusted_values = (im2[obj_pixels] * scale_factor).clip(0, 255).astype(np.uint8)
        print(adjusted_values)
        result[obj_pixels] = adjusted_values

        # Get the replaced objects (as verification)
        replaced_obj[obj_pixels] = im1[obj_pixels]
        print(adjusted_values)

    return result, replaced_obj


def data_thr_guess(
    data, channel, frames=-1, bin_value=1, fig_ncol=2, ofname=None, **kwargs
):
    """
    It uses OTSU Threshold as first approximation to define the roi binarization.
    use findContours to catch the colonies and get a more accurate center aproximation.
    it displays the binarized image and the identified centroid.

    Parameters
    ----------
    rois: dict
        the dictionary with the rois.

    channel: string
        rois data channels name to use.

    frames: int
        frame number to use.

    bin_value: int
        binary value to assign to the pixels that surpass the threshold.

    fig_ncol: int
        number of columns in the figure.

    ofname: string
        if given, the figure is stored under that name. More options
        with **kwargs  (see flua.save_fig())

    Returns
    -------
    thresholds: dict
        the Otsu threshold value for each roi.

    centroids: dict
        the centroid coordinates for each roi.
        [yc,xc]

    """
    centroids = {}
    thesholds = {}
    binims = {}
    near_contours = {}

    ## init the figure
    nrois = len(rois.values())
    nrows = nrois // fig_ncol + 1

    fig, axs = plt.subplots(
        nrows, fig_ncol, figsize=(4 * fig_ncol, 4 * nrows), layout="constrained"
    )

    if nrows > 1 and fig_ncol > 1:
        axs = axs.flatten()

    if nrows == 1 and fig_ncol == 1:
        axs = [axs]

    idx = 0  # index for the figure

    # get the roi channel data for the indicated frame
    for roi in rois.values():

        roi_id = roi.id

        roi_frame, frame = get_roi_frame(roi, channel, frame)

        # Otsu thresholding
        thr_otsu, bin_frame = cv2.threshold(
            roi_frame, 0, bin_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        thesholds[roi_id] = thr_otsu
        binims[roi_id] = bin_frame  # probably this dictionary is not necessary

        # Find the contours of the binarized image
        contours, hierarchy = cv2.findContours(
            bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # get the nearest contour to the roi center
        near_contour, centroid = get_nearest_contour(roi, contours)

        near_contours[roi_id] = near_contour
        centroids[roi_id] = centroid  # [yc,xc]

        # plot the binarized image an its centroid

        # axs[idx].imshow(roi_frame)
        axs[idx].imshow(bin_frame)
        if centroid is not None:
            axs[idx].plot(centroid[1], centroid[0], "or", linewidth=2, label="centroid")
        if near_contour is not None:
            axs[idx].plot(
                near_contour[:, 0, 0],
                near_contour[:, 0, 1],
                "r",
                linewidth=2,
                label="contour",
            )

        axs[idx].set_title(f"Colony {roi.id}, thr = {thr_otsu}, frame{frame}")
        axs[idx].axis(
            "image"
        )  # Ajusta los ejes para que coincidan con los de la imagen
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])  # Ocultar los ticks de los ejes
        axs[idx].legend(loc="lower left")

        idx += 1

    while idx < len(axs):
        axs[idx].axis("off")
        idx += 1
    plt.show()

    # save the figure file if indicated
    if ofname is not None:
        ofname += f"{frame}"
        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        fig_kwargs.update(kwargs)  # join with the input functions kwargs
        save_figure(fig, ofname, **fig_kwargs)

    return centroids, thesholds


def thr_centroid_guess(
    rois, channel, frame=-1, bin_value=1, fig_ncol=4, ofname=None, **kwargs
):
    """
    It uses OTSU Threshold as first approximation to define the roi binarization.
    use findContours to catch the colonies and get a more accurate center aproximation.
    it displays the binarized image and the identified centroid.

    Parameters
    ----------
    rois: dict
        the dictionary with the rois.

    channel: string
        rois data channels name to use.

    frame: int
        frame number to use.

    bin_value: int
        binary value to assign to the pixels that surpass the threshold.

    fig_ncol: int
        number of columns in the figure.

    ofname: string
        if given, the figure is stored under that name. More options
        with **kwargs  (see flua.save_fig())

    Returns
    -------
    thresholds: dict
        the Otsu threshold value for each roi.

    centroids: dict
        the centroid coordinates for each roi.
        [yc,xc]

    """
    centroids = {}
    thesholds = {}
    binims = {}
    near_contours = {}

    ## init the figure
    nrois = len(rois.values())
    nrows = nrois // fig_ncol + 1

    fig, axs = plt.subplots(
        nrows, fig_ncol, figsize=(4 * fig_ncol, 4 * nrows), layout="constrained"
    )

    if nrows > 1 and fig_ncol > 1:
        axs = axs.flatten()

    if nrows == 1 and fig_ncol == 1:
        axs = [axs]

    idx = 0  # index for the figure

    # get the roi channel data for the indicated frame
    for roi in rois.values():

        roi_id = roi.id

        roi_frame, frame = get_roi_frame(roi, channel, frame)

        # Otsu thresholding
        thr_otsu, bin_frame = cv2.threshold(
            roi_frame, 0, bin_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        thesholds[roi_id] = thr_otsu
        binims[roi_id] = bin_frame  # probably this dictionary is not necessary

        # Find the contours of the binarized image
        contours, hierarchy = cv2.findContours(
            bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # get the nearest contour to the roi center
        near_contour, centroid = get_nearest_contour(roi, contours)

        near_contours[roi_id] = near_contour
        centroids[roi_id] = centroid  # [yc,xc]

        # plot the binarized image an its centroid

        # axs[idx].imshow(roi_frame)
        axs[idx].imshow(bin_frame)
        if centroid is not None:
            axs[idx].plot(centroid[1], centroid[0], "or", linewidth=2, label="centroid")
        if near_contour is not None:
            axs[idx].plot(
                near_contour[:, 0, 0],
                near_contour[:, 0, 1],
                "r",
                linewidth=2,
                label="contour",
            )

        axs[idx].set_title(f"Colony {roi.id}, thr = {thr_otsu}, frame{frame}")
        axs[idx].axis(
            "image"
        )  # Ajusta los ejes para que coincidan con los de la imagen
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])  # Ocultar los ticks de los ejes
        axs[idx].legend(loc="lower left")

        idx += 1

    while idx < len(axs):
        axs[idx].axis("off")
        idx += 1
    plt.show()

    # save the figure file if indicated
    if ofname is not None:
        ofname += f"{frame}"
        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        fig_kwargs.update(kwargs)  # join with the input functions kwargs
        save_figure(fig, ofname, **fig_kwargs)

    return centroids, thesholds


def thr_screening(
    rois,
    col_id,
    thr_channel,
    frame_id,
    thr_values,
    bin_value=1,
    ncols=4,
    plot_channels=None,
    f_size=(12, 12),
    ofname=None,
    **kwargs,
):
    """
    To perform a screening of posible threshold values to define the colonies
    border.

    Parameters
    ----------
    rois: dict
        the dictionary with the rois.

    col_id: int
        colony id in the dictionary of the roi to be used.

    thr_channel: str (or the used channel key type)
        channel to be used to get the contours

    frame_id: integer
        frame id to be used

    thr_values: list or np.array
        list with threshold to be tested

    bin_value: numeric (positive)
        binary value used in cv2.threshold(). The pixels over the threshold
        are assigned this values at create the binarized image.
        by default binary_value = 1 to simply the computations

    ncols: integer
        number of columns in the output figure.
        This values is used to determine the required number of row in the figure
        to plot all the images.

    plot_channels: dictionary
        By default is None and the contours are plot over the same thr_channel
        used to get the contours.
        If given, it should have the positions and channels used to compose the
        roi image:
        e.g. { 0:'wR', 1:'wG',2:'wB'}

    f_size: tuple of integers
        figure size tuple. e.g. (12,12)

    ofname: string
        if given, the figure is stored under that name. More options
        with **kwargs  (see flua.save_fig())

    Returns
    -------
    contour_list: list
        list with the center near contour obtained with each threshold

    thresh_list: list
        list with the actual used thresholds to obtains the contours in
        contour_list.

    """
    if bin_value < 0:
        print("\nbin_value have to be positive. Default value of 1 will be used.\n")
        bin_value = 1

    # get the requested roi
    roi = rois[col_id]

    # get roi frame
    roi_im, frame_id = get_roi_frame(roi, thr_channel, frame_id)

    # init the output lists

    contour_list = []
    thresh_list = []
    # bins_list = []
    # centroids = []

    # compute the max binarized ima sum value (used to stop the computations if reached)
    max_bin_sum = roi_im.size * bin_value

    # use the threholds to get the contours
    for thr in thr_values:
        # Simple thrsholding

        out_tresh, bin_frame = cv2.threshold(roi_im, thr, bin_value, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # get the nearest contour to the roi center
        near_contour, centroid = get_nearest_contour(roi, contours)

        # Store the cicle values
        thresh_list.append(out_tresh)
        contour_list.append(near_contour)
        # bins_list.append(bin_frame)
        # centroids.append(centroid)            #[yc,xc]

        # in case each pixel of binarized image is under the threshold
        if bin_frame.max() == 0:
            print(f"Stops because all the pixels are under current threshold = {thr}")
            break

        # in case all the pixels are over the threshold
        if bin_frame.sum() == max_bin_sum:
            print(f"Stops because all the pixels are under current threshold = {thr}")
            break

    nrows = int(np.ceil(len(contour_list) / ncols))

    # if plot channels were indicated, recompose the roi image using them
    if plot_channels is not None:

        print("\nCompose image\n")

        roi_im = compose_image(roi.data, channels=plot_channels, data_frame=frame_id)  #
        # roi_im = rois[col_id].compose_image(channels = { 0:'wR', 1:'wG',2:'wB'}, data_frame = frame)

    fig, ax = plt.subplots(nrows, ncols, figsize=f_size, layout="constrained")

    n_axes = ncols * nrows

    for i in range(len(contour_list)):
        contour = contour_list[i]
        col = i % ncols
        row = i // ncols

        ax[row, col].imshow(roi_im)
        ax[row, col].plot(contour[:, 0, 0], contour[:, 0, 1], "r", linewidth=1)
        ax[row, col].set_title(f"thr = {thresh_list[i]}")
        ax[row, col].axis("off")

    # make the remaining axis off
    while i < n_axes:
        ax[i // ncols, i % ncols].axis("off")
        i += 1

    # save the figure file if indicated
    if type(ofname) is str:
        ofname += f"_col{col_id}_{frame_id}"  # f'thr_screening_col9_288'

        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        fig_kwargs.update(kwargs)  # join with the input functions kwargs
        save_figure(fig, ofname, **fig_kwargs)

    return (contour_list, thresh_list)


def get_contour(
    rois,
    channel,
    threshold,
    colonies_mask=None,
    frame=-1,
    bin_value=1,
    display=False,
    fig_ncol=4,
    return_all=False,
    ofname=None,
    **kwargs,
):
    """
    bin_value: int
        binary value to use in the thresholding

    fig_ncol: int
        number of columns in the figure

    ofname: str
        output file name

    return_all: bool
        if True, returns the contours, the centroids, and  the binarized images
        the of the ROIs

    Returns
    -------
    The keys of all the outputs dictionaries are the colonies ids.

    rois_contours: dict
        dictionary with the contour of each colony.

    [OPTIONALS]

    rois_centroids: dict
        dictionary with the centroid of each colony.

    rois_binims: dict
        dictionary with the binarized image of each colony.


    """
    # init the storage dictionaries
    rois_binims = dict()
    rois_centroids = dict()
    rois_contours = dict()

    # Mask all the ROIs if indicated
    if colonies_mask is not None:

        if display == True:
            ## init the figure
            nrois = len(rois.values())
            nrows = nrois // fig_ncol + 1

            fig, axs = plt.subplots(
                nrows, fig_ncol, figsize=(4 * fig_ncol, 4 * nrows), layout="constrained"
            )

            if nrows > 1 and fig_ncol > 1:
                axs = axs.flatten()

            if nrows == 1 and fig_ncol == 1:
                axs = [axs]

            idx = 0  # index for the figure

        for roi in rois.values():
            # Get the mask for the ROI
            mask = colonies_mask[f"colony_{roi.id+1}"][mask] > 0
            mask = mask > 0
            # Crop the mask to the ROI size
            xlims = roi.xlims
            ylims = roi.ylims
            mask = mask[ylims[0] : ylims[1] + 1, xlims[0] : xlims[1] + 1]
            # Perform a binary dilation using OpenCV to increase the mask area
            kernel = disk(3)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=3)
            # Apply the mask to the ROI frame
            roi_frame, frame = get_roi_frame(roi, channel, frame)
            roi_frame = roi_frame * mask
            # roi_frame, frame = get_roi_frame(roi, channel, frame)
            # Simple thrsholding

            out_tresh, bin_frame = cv2.threshold(
                roi_frame, threshold, bin_value, cv2.THRESH_BINARY
            )
            contours, hierarchy = cv2.findContours(
                bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            # get the nearest contour to the roi center
            near_contour, centroid = get_nearest_contour(roi, contours)

            # Store the cicle values
            rois_contours[roi.id] = near_contour
            rois_centroids[roi.id] = centroid  # [yc,xc]
            rois_binims[roi.id] = bin_frame

            # Display the figure results if indicated
            if display == True:
                roi_im, _ = get_roi_frame(roi, channel, frame)
                # axs[idx].imshow(roi_frame)
                # axs[idx].imshow(bin_frame)
                axs[idx].imshow(roi_im)
                axs[idx].plot(
                    centroid[1], centroid[0], "or", linewidth=2, label="centroid"
                )
                if near_contour is not None:
                    axs[idx].plot(
                        near_contour[:, 0, 0],
                        near_contour[:, 0, 1],
                        "r",
                        linewidth=2,
                        label="contour",
                    )

                axs[idx].set_title(f"Colony {roi.id}, thr = {threshold }, frame{frame}")
                axs[idx].axis(
                    "image"
                )  # Ajusta los ejes para que coincidan con los de la imagen
                axs[idx].set_xticks([])
                axs[idx].set_yticks([])  # Ocultar los ticks de los ejes
                axs[idx].legend(loc="lower left")

                idx += 1

        if display == True:
            while idx < len(axs):
                axs[idx].axis("off")
                idx += 1
            plt.show()

        # save the figure file if indicated
        if ofname is not None:
            ofname += f"{frame}_thr{threshold:03}"
            fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
            fig_kwargs.update(kwargs)  # join with the input functions kwargs
            save_figure(fig, ofname, **fig_kwargs)

        if return_all == True:
            return (rois_binims, rois_centroids, rois_contours)
        else:
            return rois_contours

    else:

        # Print that we expect a mask to be used
        print(
            "No mask was provided. Please provide a mask to use for the colonies segmentation."
        )
        return None


def get_all_contours(rois, channel, threshold, frames=None, bin_value=1):
    """

    frames: list
        list of frames which contour will be obtained. If None, all the frames are used.

    bin_value: int
        binary value to use in the thresholding

    fig_ncol: int
        number of columns in the figure

    ofname: str
        output file name

    return_all: bool
        if True, returns the contours, the centroids, and  the binarized images
        the of the ROIs

    Returns
    -------
    The keys of all the outputs dictionaries are the colonies ids.

    rois_contours: dict
        dictionary with the contour of each colony.

    [OPTIONALS]

    rois_centroids: dict
        dictionary with the centroid of each colony.

    rois_binims: dict
        dictionary with the binarized image of each colony.


    """
    # init the storage dictionaries
    rois_contours = dict()
    rois_centroids = dict()
    rois_areas = dict()

    # Loop over the ROIs
    for roi in rois.values():

        print(roi.id)
        # check the dimentions
        if roi.data[channel].ndim == 3:

            ## init the list of the ROI valies in the dictionaries
            rois_contours[roi.id] = list()
            rois_centroids[roi.id] = list()
            rois_areas[roi.id] = list()

            # define the frames if were not specified
            if frames is None:

                nframes = roi.data[channel].shape[2]
                frames = np.arange(0, nframes)

            # go for each frame
            for frame in frames:
                roi_frame, _ = get_roi_frame(roi, channel, frame)

                # Simple thrsholding
                out_tresh, bin_frame = cv2.threshold(
                    roi_frame, threshold, bin_value, cv2.THRESH_BINARY
                )
                contours, _ = cv2.findContours(
                    bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                # get the nearest contour to the roi center
                near_contour, centroid = get_nearest_contour(roi, contours)

                try:
                    area = cv2.contourArea(near_contour)
                except:
                    area = 0

                # append the cicle values
                rois_contours[roi.id].append(near_contour)
                rois_centroids[roi.id].append(centroid)  # [yc,xc]
                rois_areas[roi.id].append(area)
        else:

            print(f"ROI {roi.id} has only one frame")
            roi_frame = get_roi_frame(roi, channel, -1)

            # Simple thrsholding
            out_tresh, bin_frame = cv2.threshold(
                roi_frame, threshold, bin_value, cv2.THRESH_BINARY
            )
            contours, _ = cv2.findContours(
                bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            # get the nearest contour to the roi center
            near_contour, centroid = get_nearest_contour(roi, contours)

            try:
                area = cv2.contourArea(near_contour)
            except:
                area = 0

            # append the cicle values
            rois_contours[roi.id] = near_contour
            rois_centroids[roi.id] = centroid  # [yc,xc]
            rois_areas[roi.id] = area

    return (rois_contours, rois_centroids, rois_areas)


def border_signal(roi, wide=1, offset=1, time_key="W"):
    """
    To compute the mean and std of the colony border signal over time.
    It is the dynamic border that moves as colony growths

    Parameters
    ----------

    wide: numeric
        wide of the border band

    offset: numeric
        distance to skip from the border

    time_key: str
        key of the time serie to use from rot.times

    Returns
    -------
    mean_bband: 1d array
        mean signal of the border band

    std_bband: 1d array
        std signal of the border band

    """

    # init the list to store the band value
    mean_bband = np.zeros_like(roi.times[time_key])
    std_bband = np.zeros_like(roi.times[time_key])

    # go for each frame
    for frame in range(len(mean_bband)):
        dms_values = roi.rms[frame]  # [distances, averages, stds]
        dists = dms_values[0]
        signal = dms_values[1]

        try:
            up_limit = dists.max() - np.abs(offset)

        except:
            up_limit = 0

        low_limit = up_limit - wide

        # get the indices of the radius
        indices = (dists >= low_limit) & (dists <= up_limit)
        mean_band_frame = signal[indices].mean()
        std_band_frame = signal[indices].std()

        # assign to the arrays
        mean_bband[frame] = mean_band_frame
        std_bband[frame] = std_band_frame

    return (mean_bband, std_bband)


def sort_elements(elements, gybs):
    """
    This function is designed to sort the identified elements based on their
    position. "elements" have to be a list of lists or arrays where the first
    two values are "y" and "x" position respectively (it doesn't matter the number
    of other values).
    e.g: elements = [[y1,x1, r1,..],[y2,x2, r2,..],...[yn,xn,rn,..]]
    Then, their are sorted in descendent rows defined by the boundaries indicated
    in "gybs" list, and from left to rigth on each that rows.

    Algorithm:
    1) elements are sorted by their y-axis position.
    2) grouped accord the y axis boundaries indicated in gybs
    3) sorted inside each group by their x-axis position
    4) joined all together in descendat groups order

    Parameters
    ----------
    elements : list or np.array
        list of elements where the first two values are "y" and "x" position respectively
        e.g: elements = [[y1,x1, r1,..],[y2,x2, r2,..],...[yn,xn,rn,..]]

    gybs : list
        y groups boundaries.

    Returns
    --------
    xysort : list
        list of sorted elements

    """

    # sort the elements based on the y axis position
    ysort = sorted(elements, key=lambda x: x[0])

    ##################
    #### sort them based on x axis position in each of the boudaries groups

    # split them in y boundaries groups
    groups = dict()  # dictionary to temporary store the groups

    for e in ysort:
        yval = e[0]

        for j in range(len(gybs) + 1):

            # define the group baundaries
            if j == 0:
                binf = 0  # zero for the first group

            else:
                binf = gybs[j - 1]  # inferior group boundary

            try:
                bup = gybs[j]  # up group boundary

            except:
                bup = np.inf

            # classify the element and store it in the dictionary
            if yval > binf and yval < bup:

                try:
                    groups[j].append(e)

                except:
                    groups[j] = list()
                    groups[j].append(e)

    ## sort each group and join them again
    xysort = list()
    for key in list(groups.keys()):
        g_current = groups[key]

        # sort the current group
        xsort = sorted(g_current, key=lambda x: x[1])

        # pull they elements to the final sorted list
        for e in xsort:
            xysort.append(e)

    # return the sorted elements
    return xysort


def get_roi(rois, col_id, not_found_return=False):
    """
    get a ROI from a dictionary or list of ROIs based on the ROI id number.
    The key of the dictionary are expected to be the id numbers, on the contrary
    is display some warning.

    If you input a single ROI, it can be used to verify its ID. If it doesn´t
    match the input col_id, the not_found_return will be returned.

    """

    if type(rois) == dict:
        try:
            roi = rois[col_id]

            # verify it is the same
            if roi.id == col_id:

                return roi

            else:
                for roi in rois.values():
                    if roi.id == col_id:
                        print(
                            f"\nNotice: the roi id number ({col_id}) doesn't match its dictionary key"
                        )
                        return roi

                # in case it is not found.
                print(
                    f"\nCannot found the roi {col_id}. Verify input ROIs and/or colony id"
                )
                return not_found_return

        # in case col_id is not in the dictionary keys
        except:

            for roi in rois.values():
                if roi.id == col_id:
                    print(
                        f"\nNotice: the roi id number ({col_id}) doesn't match its dictionary key"
                    )
                    return roi

            # in case it is not found.
            print(
                f"\nCannot found the roi {col_id}. Verify input ROIs and/or colony id"
            )
            return not_found_return

    # in case it is directly a ROI, verify if it is the indicated col_id
    elif type(rois) == ROI:

        if roi.id == col_id:
            return roi

        else:
            print(
                f"The ID of the input ROI is {roi_i.id}, and don't match the requested id"
            )
            return not_found_return

    elif type(rois) == list or type(rois) == tuple:
        # search in the list
        for roi in rois:
            if roi.id == col_id:
                return roi
        # in case it is not found.
        print(f"\nCannot found the roi {col_id}. Verify input ROIs and/or colony id")
        return not_found_return

    else:
        raise Exception('\nInvalid "rois variable" type input.')


def obtain_rois(
    data,
    blobs,
    channels=CHANNELS,
    rfactor=1.1,
    sc_suffix=["", "c"],
    chan_descript="",
    square_roi=True,
    circular_roi=False,
    frames=...,
):
    """
    Based on the information of each identified colony, create arrays to contain
    the regions of interest (ROI) around each one.
    The distance that ROI comprise from the center is determined by
    rfactor*blob_radius.
    Use the factor to take a bit more area around the colony.

    Parameters
    ----------
    data: dictionary
        It contains information of the images separated in channels
        Structure: data[channel][n,m,...], usually [n,m,frames]

    blobs: array (Nx3)
        contains the (y,x) center position and radius of each blob
        for each of N colonies.
        This values should be relative to the input data.
        Structure for the 'i' element: [yi, xi, ri]

    channels: dict or list
        For ths functions, the positions are not used.
        dictionary with the channels positions and names.
        e.g. channels = { 0 : 'R', 1 : 'G'}, indicate the image has the channels
        R and G inpositions 0 and 1 respectivelly.

    rfactor: numeric
        The factor applied to  blob radius to define the ROI size as a proportion
        of the radius.
        The ROI is extended from the center of blob to rfactor*blob_radius.

    sc_suffix = : list
        output channels names modifiers. It is a list of two strings wich are
        added to the end of the ones in input channels.
        The first is added at the end of the square rois, and the second to the
        circular rois.

    chan_descript: str
        Some extra text description for the output channels. There is a basic default
        description that will be added anyways.

    square_roi : bool
        if True, the square ROI is computed and obtained

    circular_roi: bool
        if True, the circular ROI is computed and obtained

    frames: interger or list
        to obtain specific frames from the data
        default is ... (which means all)

    Returns
    -------
    rois: dict
        With the square or circular ROIS for each colony in each input channel.
        estructure: rois[blob_id]['channel_name'][n,m,...]  (tipcally [y,x, frames])
        Accord the output values correspond to square or circular neighbourhood,
        there are two groups of outpurt channels:
        Fist group:
            The ROI array image data for square region around colony position
            of side  factor*(colony radius)
        Second group:
            The ROI array image data only within circle (radius = width/2),
            with the data outside the circle equal to zero.
            The size of the array is equal to square ROIS (all_rois) size.

    """

    if type(channels) == dict:
        channels = list(channels.values())

    rois = dict()

    nc = len(blobs)

    # go for each colony
    for i in range(nc):

        # to store the channels data for the colony i
        data_ic = dict()

        # whole image shape should be the same for all the data channels of the colony
        h, w = data[channels[0]].shape[0:2]

        # Get the blob center and radius
        yi = blobs[i, 0]
        xi = blobs[i, 1]
        ri = blobs[i, 2]  # blob radius

        r_roi = rfactor * ri  # ROI radius for square and circular

        # compute ROI limits
        x = np.array([xi - r_roi, xi + r_roi])
        y = np.array(
            [yi - r_roi, yi + r_roi]
        )  # +1 will be added directly at the moment of perform the slicing

        ## get the x,y center of each ROI/colony ##
        # xc = (x[1]-x[0]-1)/2
        xc, yc = (
            r_roi,
            r_roi,
        )  # doesn't need to be integers. better float to calculate circle
        # center is ROI[r,r]    #taking into account that starts from 0.

        # correct the out of image boundary cases (colonies at the borders of the image)
        # but keeps the colony center coordinates.

        xwr = x[1] - w  # right
        if xwr > 0:
            x[1] = w

        if x[0] < 0:  # left
            xc += x[0]  # reduce xc
            x[0] = 0

        yhb = y[1] - h  # bottom
        if yhb > 0:
            y[1] = h

        if y[0] < 0:  # up
            yc += y[0]  # reduce yc
            y[0] = 0
        # --> effects on colony center are just for up or left (y[0] or x[0] ) border reduction

        # init channels descriptors dictionary
        c_descriptors = {}

        for chani in channels:

            # define the output channels names
            name_s = chani + sc_suffix[0]  # for square ROIs
            name_c = chani + sc_suffix[1]  # for circular ROIs

            # get square ROI data for the current channel
            try:

                # round the limits values to make them integers
                y0, y1 = round_up(y[0]), round_up(y[1])
                x0, x1 = round_up(x[0]), round_up(x[1])
                print(f"ROI {i}: xlims = [{x0},{x1}], ylims = [{y0},{y1}]")

                # get the ROI
                sq_roi = data[chani][
                    y0 : y1 + 1, x0 : x1 + 1, frames
                ]  # +1 because slicing sintaxis

            except:
                print(f"ROI {i} of channel {chani} fail to be obtained")
                sq_roi = None

            # if square roi was indicated to obtain
            if square_roi:
                data_ic[name_s] = sq_roi
                c_descriptors[name_s] = f"Square ROI from {chani}. " + chan_descript

            # if circular roi was indicated to obtain
            if circular_roi:
                try:
                    # get the roi heigh and wide
                    hi, wi = sq_roi.shape[0:2]

                    # get the roi data type
                    # roi_dtype = sq_roi.dtype

                    # obtain the circular ROI for this channel
                    # c_roi = np.zeros((sq_roi.shape), dtype = roi_dtype)
                    c_roi = np.zeros_like(sq_roi[:, :, frames])

                    # take just the values that are inside the colony circle
                    # left the rest in zero.
                    # for n in range(wi):
                    #    for m in range(hi):
                    #        if ((n-xc)**2+(m-yc)**2) <= r_roi**2:
                    #            c_roi[m,n,...] = sq_roi[m,n,frames]

                    # Generate sparse coordinate grids
                    ogy, ogx = np.ogrid[:hi, :wi]
                    # get the mask of index inside the circle
                    mask = (ogx - xc) ** 2 + (ogy - yc) ** 2 <= r_roi**2
                    # apply the mask
                    c_roi[mask, ...] = sq_roi[mask, frames]

                except:
                    print(f"Circular ROI {i} of channel {chani} fail to be obtained")
                    c_roi = None

                data_ic[name_c] = c_roi
                c_descriptors[name_c] = f"Circular ROI from {chani}. " + chan_descript

        # to store the integer values of limits
        y = np.array([y0, y1])
        x = np.array([x0, x1])

        # create and store the roi object
        rois[i] = ROI(
            i, data_ic, x, y, blobs[i, :], center=[yc, xc], descriptors=c_descriptors
        )

        # store the rfactor in the dataset.

    return rois


def blobs_translation(blobs, new_centers, old_rois, print_diff=True):
    """
    To translate the blobs coordinates based in center translation of old_rois
    blobs values are defined in the coordinates of the original image
    new_centers are defined in the coordinates of each related roi (a subpart of the whole image)


    blobs rows id should correspond to each new_centers key.
    e.g. blobs[i] should correspond with new_centers[i]

    Parameters
    ----------
    blobs: array (Nx3)
        contains the (y,x) center position and radius of each blob
        for each of N colonies.
        Structure for the 'i' element: [yi, xi, ri]

    new_centers: dict
        it contain a list for eac new center with the coordinates of them.
        Their values are refered to the ROI coordinates (not to the original whole image)
        e.g. {id1 : [cy1, cx1], ..., idn [cyn, cxn]}

    old_rois: list
        list with the ROI objects wich new_centers were defined.
        The difference between new_centers and the centers of old_rois
        are used to modify the centers of blobs.

    print_diff: boolean
        If True, the center diference is printed

    Returns
    -------
    t_blobs: array (Nx3)
        blobs with translated values. Same structure as input blobs element.
        its coordinates are refered to the original image.

    """

    # init the new array
    t_blobs = np.zeros(blobs.shape)

    # go for each blob
    for i in range(len(blobs)):

        # get the old roi centers
        rcy = old_rois[i].center[0]
        rcx = old_rois[i].center[1]

        # get the new roi center values
        ncy = new_centers[i][0]
        ncx = new_centers[i][1]

        # get the old blob values related to the whole image
        blob = blobs[i]

        b_cy = blob[0]
        b_cx = blob[1]
        b_r = blob[2]

        # compute the diference between the roi centers
        dy = ncy - rcy
        dx = ncx - rcx

        # remap the new center in the whole image
        t_cy = b_cy + dy
        t_cx = b_cx + dx

        # round the values
        t_cy = int(np.floor(t_cy + 0.5))
        t_cx = int(np.floor(t_cx + 0.5))

        # store the values
        t_blobs[i, :] = [t_cy, t_cx, b_r]

        if print_diff:
            print(
                f"[{i:{2}}] move [{int(np.floor(dy + 0.5)):{2}},{int(np.floor(dx + 0.5)):{2}}]"
            )

    return t_blobs


def load_rois_data(
    rois,
    fpaths,
    channels=CHANNELS,
    pfx_sfx=["", ""],
    sc_suffix=["", "c"],
    square_roi=True,
    circular_roi=False,
    chans_times={},
    chans_descriptions={},
    **kwargs,
):
    """
    To load just the ROIs and not the whole files.
    If you are loading different ROIs for the same data, its more efficient to
    use this function instead of the individual ROI class method, because it reads
    the images just one time for all the input ROIs.

    Make sure to use prefix and suffix propperly to avoid deletion of others channels.

    Parameters
    ----------
    rois: dict
        dict of ROIs, with the IDs as keys

    fpaths: list of string
        list with the name of the files to be added

    channels: dict
        with the position and name of the channels

    pfx_sfx: list of strings
        prefix and suffix added to each of the given channels to
        store them in the ROI.
        structure: [preffix, suffix]

    c_suffix: list of strings
        extra suffix added to each of the given channels to differenciate between
        square and circular ROIs if requested.
        structure: [square_suffix, circular_suffix]

    chans_times: dict
        to map channels to a time serie stores in self.times
        e.g. {'chan1': 'times3', 'chan2': 'times1', 'chan4': 'times1'}
        channels not included in the dict will be assigned None.

    chans_descriptions: dict
        extra text to detail the channel content.
        dictionary with structure:  {channel : 'descriptor text'}
    """

    # built the output channels by adding the preffix and suffix to every input channel
    ochans = [pfx_sfx[0] + chani + pfx_sfx[1] for chani in channels.values()]

    # check the new channels are not previously stored in any input roi
    for roi in rois.values():

        sc_ochans = list()

        for ochani in ochans:
            if square_roi == True:
                sc_ochans.append(ochani + sc_suffix[0])

            if circular_roi == True:
                sc_ochans.append(ochani + sc_suffix[1])

        if roi.check_channels(sc_ochans, ask=True):
            print(
                "\nThe data was not loaded because some channel names are the same as preexisting ones. "
                "Please change the output channel names by adding preffix ans suffixs."
            )
            return ()

    # read the data files
    data, chans = get_im_data(
        f_names=fpaths, channels=channels, pfx_sfx=pfx_sfx, **kwargs
    )

    add_rois_data(
        rois,
        data,
        chans,
        sc_suffix,
        square_roi,
        circular_roi,
        chans_times=chans_times,
        chans_descriptions={},
    )

    return


def add_rois_data(
    rois,
    data=None,
    channels=CHANNELS,
    sc_suffix=["", "c"],
    square_roi=True,
    circular_roi=False,
    chans_times={},
    chans_descriptions={},
):
    """
    add channels to input rois, using input data and the comprised ROI area
    indicated in each ROI object of rois.
    It cab be requested square and/or circular ROIs
    Parameters
    ----------

    data:
        if given, the rois will be obtained from there
        To get circular ROIS from pre-existing  use None.
        and input the list desired ROIs channels.

    sc_suffix : list
        list with the suffix for the output channels names.
        the first is the suffix for used for square ROIs
        the second is the suffix for used for circular ROIs

    square_roi : bool
        if True, the square ROI is computed and obtained

    circular_roi: bool
        if True, the circular ROI is computed and obtained

    chans_times: dict
        relates channels to times series data
        chan_times.keys() has to be the same as input channels.

    chans_descriptions: dict
        Some extra text description for each of the output channels.
        There is a default basic description that will be added anyways.
        chan_drescription.keys() has to be the same as input channels.

    """
    ## get channel names
    if type(channels) == dict:
        channels = list(channels.values())

    ## in case of no define them, create the empty dictionaries to avoid errors.
    if chans_times == {}:
        for chan in channels:
            chans_times[chan] = None

    if chans_descriptions == {}:
        for chan in channels:
            chans_descriptions[chan] = ""

    ## go for each ROI
    for roi in rois.values():

        # get the roi coordinate limits inside the data
        y = roi.ylims
        x = roi.xlims

        # and its center coordinates in the source image and radius
        yc = roi.blob[0]
        xc = roi.blob[1]
        rc = roi.rroi  # roi radius (not colony radius), used to get circular rois.

        # init the output channels descriptors and channels to times relation dictionaries
        oc_descriptors = {}
        ochans_times = {}

        # obtain the values of each indicated data channel
        for chani in channels:

            # the ciruclar ROIs are made from square ROIs, if you only want the circular, Squares will be obtained temporary
            do_square_roi = False  # start the token
            if circular_roi == True:
                # in case there is no previous square rois called chani
                if chani not in roi.channels:
                    # do square roi first, but dont save it at least square_roi is True too
                    do_square_roi = True

            # get square ROI data for the current channel from input data
            if (square_roi == True or do_square_roi) and data is not None:

                # define the output channels names
                ochani = chani + sc_suffix[0]

                try:

                    # round the limits values to be sure they are integers
                    y0, y1 = round_up(y[0] + 0.5), round_up(y[1] + 0.5)
                    x0, x1 = round_up(x[0] + 0.5), round_up(x[1] + 0.5)
                    print(f"ROI {roi.id}: xlims = [{x0},{x1}], ylims = [{y0},{y1}]")

                    # get the ROI
                    sq_roi = data[chani][
                        y0 : y1 + 1, x0 : x1 + 1, ...
                    ]  # add 1 because the slicing

                except:
                    print(f"Square ROI {roi.id} of channel {chani} fail to be obtained")
                    sq_roi = None

                # add in case it was requested
                if square_roi:

                    roi.data[ochani] = sq_roi

                    # add the information to the descriptor and channel time relation
                    oc_descriptors[ochani] = (
                        f"Square ROI from {chani}. " + chans_descriptions[chani]
                    )
                    ochans_times[ochani] = chans_times[chani]

                    print(f"\nChannel {ochani} of added succesfully to ROI {roi.id} \n")

            # get circular ROI data for the current channel
            if circular_roi:

                # define the output channels names
                ochani = chani + sc_suffix[1]

                try:

                    # in case chani was previosly in rois
                    if do_square_roi is False:

                        sq_roi = roi.data[chani]

                    # get the roi heigh and wide
                    hi, wi = sq_roi.shape[0:2]

                    # get the roi data type
                    # roi_dtype = sq_roi.dtype

                    # obtain the circular ROI for this channel
                    # c_roi= np.zeros((sq_roi.shape), dtype = roi_dtype )
                    c_roi = np.zeros_like(sq_roi)

                    # take just the values that are inside the colony circle
                    # left the rest in zero.
                    # for n in range(wi):
                    #    for m in range(hi):
                    #        if ((n-xc)**2+(m-yc)**2) <= rc**2:
                    #            c_roi[m,n,...] = sq_roi[m,n,...]

                    # Generate sparse coordinate grids
                    ogy, ogx = np.ogrid[:hi, :wi]
                    # get the mask of index inside the circle
                    mask = (ogx - xc) ** 2 + (ogy - yc) ** 2 <= rc**2
                    # apply the mask
                    c_roi[mask, ...] = sq_roi[mask, ...]

                except:
                    print(
                        f"Circular ROI {roi.id} of channel {chani} fail to be obtained."
                    )
                    c_roi = None

                ## add the values
                roi.data[ochani] = c_roi

                # add the information to the descriptor and channel time relation
                oc_descriptors[ochani] = (
                    f"Circular ROI from {chani}. " + chans_descriptions[chani]
                )
                ochans_times[ochani] = chans_times[chani]

                print(f"Channel {ochani} added succesfully to ROI {roi.id} \n")

        # update the roi channels
        roi.update_channels(ochans_times, oc_descriptors)

    return


def generate_control_signal(control_regimen, times, time_key="T"):
    """
    Generate a control signals vector accord the input times vector

    control_regimen: dict
        it contains the control regimen schedule information. Espected estructure:
        control_regimen = {'T' : [24,48,72],
                            'R' : [0,100,0],
                            'G' : [100,0,100]}
        'T' : list or 1d array
            time limits in hours

        The other elememts are the power percentage for each time limit
        for the control channel indicated in by the key.

    times: list or 1d array
        times in hours
    time_key: str
        key of time limits in control_regimen dict
    """
    # check the type of times
    if isinstance(times, (int, float, np.number)):
        times = np.array([times])

    times = np.array(times)

    # get the regimens chedule limits
    tlims = control_regimen[time_key]

    # get the other keys asociated to control elements
    ckeys = list(control_regimen.keys())
    ckeys.remove(time_key)

    # generate the control signals
    control_signal = dict()

    for ckey in ckeys:

        control_signal[ckey] = np.zeros_like(times)

        powers = control_regimen[ckey]  # signal power at each time block

        # go for each time block
        init = 0
        for tlim, power in zip(tlims, powers):
            # get the indices of the current time bloque
            if init == 0:
                indices = (times >= init) & (times <= tlim)
            else:
                indices = (times > init) & (times <= tlim)

            control_signal[ckey][indices] = power

            # update the init
            init = tlim

    return control_signal


def fixed_bands_signal(roi, d_chan, width=None, n_bands=None, d0=0, signal_attr="rms"):
    """
    Compute the mean signal at fixed distance bands for each time. It is, take all the
    pixels inside that band and compute the mean signal of them as just one
    characteristic value of the band.
    Their are defined by specifying the number of bands or the width of each band.
    if both are defined, it will use the width.

    Note: both band limits are inclusives --> border pixels could be part of
    two diffent bands and the mean of the bands could be different than the whole colony
    mean.

    Parameters
    ----------
    roi: object
        the ROI colony object to be used

    d_chan : str
        name of the distances channel to be used to define the bands. It have to
        be the sames as the distance channel asociated to the signal.

    width : float
        the width of the band in the propper units of the distance channel.

    n_bands : int
        the number of bands to be used. The width of the bands is based on the
        maximum distance band is the same for all of them.

    d0 : float
        The lower limit of the first band. Typically it is 0.


    signal_attr : str
        name of the selected signal roi attribute

    Returns
    -------
    The dictionries keys are the band numbers relatives to the bands array.

    mean_bands_signal: dict
        the mean of the signal for each band.

    std_bands_signal: dict
        the standard deviation of the signal for each band

    bands: 1d array
        the band limits

    """

    # get the biggest colony radius
    max_dist = roi.data[d_chan].max()

    if width is not None:
        n_bands = int(max_dist / width)

    elif n_bands is not None:
        width = max_dist / n_bands

    else:

        raise ValueError("Specify the band width or the number of bands")

    # create the bands
    bands = np.linspace(d0, max_dist, n_bands + 1)

    # init the dictionaries
    mean_bands_signal = dict()
    std_bands_signal = dict()

    # get the signal
    rms = getattr(roi, "rms")
    ntimes = len(rms)  # number of times

    # go for each band
    for i in range(1, n_bands + 1):
        # define tha band limits
        dist_band = [bands[i - 1], bands[i]]

        # init the list to store the band values
        mean_sband = np.zeros(ntimes)
        std_sband = np.zeros(ntimes)

        for frame in range(ntimes):

            dms_values = rms[frame]  # [distances, averages, stds]
            dists = dms_values[0]
            signal = dms_values[1]

            # get the indices of the radius inside the band (both limits inclusives)
            indices = (dists >= dist_band[0]) & (dists <= dist_band[1])
            mean_band_frame = signal[indices].mean()
            std_band_frame = signal[indices].std()

            # assign to the arrays
            mean_sband[frame] = mean_band_frame
            std_sband[frame] = std_band_frame

        mean_bands_signal[i - 1] = mean_sband
        std_bands_signal[i - 1] = std_sband

    return (mean_bands_signal, std_bands_signal, bands)


def border_signal(roi, wide=1, offset=1, time_key="W"):
    """
    To compute the mean and std of the colony border signal over time.
    It is the dynamic border that moves as colony growths

    Parameters
    ----------

    wide: numeric
        wide of the border band

    offset: numeric
        distance to skip from the border

    time_key: str
        key of the time serie to use from rot.times

    Returns
    -------
    mean_bband: 1d array
        mean signal of the border band

    std_bband: 1d array
        std signal of the border band

    """

    # init the list to store the band value
    mean_bband = np.zeros_like(roi.times[time_key])
    std_bband = np.zeros_like(roi.times[time_key])

    # go for each frame
    for frame in range(len(mean_bband)):
        dms_values = roi.rms[frame]  # [distance, mean, std]
        dists = dms_values[0]
        signal = dms_values[1]

        try:
            up_limit = dists.max() - np.abs(offset)

        except:
            up_limit = 0

        low_limit = up_limit - wide

        # get the indices of the radius
        indices = (dists >= low_limit) & (dists <= up_limit)
        mean_band_frame = signal[indices].mean()
        std_band_frame = signal[indices].std()

        # assign to the arrays
        mean_bband[frame] = mean_band_frame
        std_bband[frame] = std_band_frame

    return (mean_bband, std_bband)


def get_mean_radial_signal(roi, s_chan, d_chan, frame):
    """
    TO get the mean radial signal of a given frame in a given roi.
    This is carried on using the inverse trandformation diatance information
    to get the radial signal asoicated to each distance.

    Paramemters
    -----------

    roi: ROI
        the roi elements to get the data
    s_chan: str
        the name of the signal channel
    d_chan: str
        the name of the distance channel
    frame: int
        the frame to get the data
    """

    # transform the arrays to 1d
    dist_flat = roi.data[d_chan][:, :, frame].flatten()
    signal_flat = roi.data[s_chan][:, :, frame].flatten()

    # get the ordered indices of the distance flatten array
    ordered_idx = np.argsort(dist_flat)

    # Order the distance and signal values array based on the ordered indices
    signal_sort_by_dist = signal_flat[ordered_idx]
    sorted_distances = dist_flat[ordered_idx]

    # get the array of observed distance values
    unique_dist = np.unique(dist_flat)  # the non-redundadnt list of distances

    mean_signal = np.zeros(unique_dist.shape)
    std_signal = np.zeros(unique_dist.shape)

    for i in range(unique_dist.shape[0]):
        dist = unique_dist[i]
        mean_signal[i] = np.mean(signal_flat[dist_flat == dist])
        std_signal[i] = np.std(signal_flat[dist_flat == dist])

    # store the radius, the mean signal of each one and its standard deviation
    outside_radial_sig = [unique_dist[-1], mean_signal[-1], std_signal[-1]]
    mean_radial_signal = [unique_dist[:-1], mean_signal[:-1], std_signal[:-1]]

    return mean_radial_signal, outside_radial_sig


def radial_mean_signal(image, center=None, r_axis=None):
    """
    Compute the mean signal for each radial distance from a point in the image.

    Parameters
    ---------
    image: array_like
        The image to be processed. It could be a monochromatic or
        multichannel image.

    center: list
        (y,x) coordinates to be used as the center of the radial average.

    r_axis: array_like
        The radial axis values.

    Returns
    -------
    averages: array_like
        The mean signal for each radial distance from the center.

    Examples
    --------
    """

    # if center is not defined, use the center of the image.
    if center is None:
        center = ((image.shape[0] - 1) // 2, (image.shape[1]) // 2)

    yc = center[0]
    xc = center[1]

    # get the y,x coordinates of each pixel.
    y, x = np.indices((image.shape[:2]))  # NxM matrix for cord_y and for cord_x.

    # compute the distance of each pixel to yc,xc
    d = np.sqrt((y - yc) ** 2 + (x - xc) ** 2)

    # round d to be able to group them in discrete groups of unitary incrase
    d = round_up(d)

    if r_axis is None:
        r_axis = np.unique(
            d
        )  # the radius value of each distance group (equal to the axis value).

    # compute the mean and std of pixel values at each radial distance group.
    mean_signal = [np.mean(image[d == di]) for di in r_axis]
    std_signal = [np.std(image[d == di]) for di in r_axis]

    # group the results
    radial_values = np.zeros((r_axis.shape[0], 3))
    radial_values[:, 0] = r_axis
    radial_values[:, 1] = mean_signal
    radial_values[:, 2] = std_signal

    return radial_values


def get_rmsf(
    roi,
    channel,
    frame=-1,
    col_id=None,
    display=False,
    color="r",
    lw=2,
    crad=2,
    ofname=None,
    fformat=".pdf",
    **kwargs,
):
    """
    get the radial mean signal of indicated colony roi channel and frame

    crad: int
        radius if the circle displayed at the colony center coordinates in the ROI
    """

    # get the roi
    roi = get_roi(roi, col_id)

    # get some colony attributes
    col_id = roi.id

    yci = round_up(roi.center[0])
    xci = round_up(roi.center[1])

    # get the ROI data channel in the indicated frame
    roi_cf = roi.data[channel][:, :, frame]

    # compute the mean radial signal of this data
    rm_signal = radial_mean_signal(roi_cf)

    rvals = rm_signal[:, 0]
    mean_rs = rm_signal[:, 1]
    std_rs = rm_signal[:, 2]

    if display:
        # Display the results
        fig, axs = plt.subplots(2, 1, figsize=(12, 5), layout="constrained")
        fig.suptitle(f"colony {col_id}, channel {channel},frame {frame}", fontsize=12)

        # Graficar los promedios radiales
        axs[0].plot(rvals, mean_rs, color="y", label="mean values")

        # add the standard deviation of them
        axs[0].fill_between(
            rvals,
            np.clip(mean_rs - std_rs, a_min=0, a_max=None),
            mean_rs + std_rs,
            color="y",
            alpha=0.2,
            label=f"std",
        )

        axs[0].set_xlabel("radial distance from colony center [px]")
        axs[0].set_ylabel("mean pixel value")
        axs[0].set_title(f"Mean radial signal")
        axs[0].legend()

        # plot the indicated ROI channel frame r
        imi = axs[1].imshow(roi_cf)

        # mark its center
        circle = plt.Circle(
            (xci, yci), radius=crad, color=color, fill=True, lw=lw, **kwargs
        )
        axs[1].add_artist(circle)
        axs[1].set_title(f"ROI")

        fig.colorbar(imi, fraction=0.035)

        plt.show()

    return rm_signal


def get_rms(
    rois,
    col_ids,
    channels,
    display=False,
    assign=False,
    t_axis_pos=None,
    t_axis_values=None,
    ofname=None,
    fformat=".pdf",
    overwrite=False,
    **kwargs,
):
    """
    get the radial mean signal of indicated(s) colony roi channel(s)

    rois: dict
        dictionary of ROIS
    col_ids: list
    channels: list

    """

    # go for each indicated colony
    for col_id in col_ids:

        roi = rois[col_id]

        for chani in channels:

            roi_c = roi.data[chani]
            nframes = roi.nframes

            # get the radial mean signal for each frame
            for frame_i in range(nframes):

                # compute the values
                roi_cfi = roi_c[:, :, frame_i]

                rm_signal_i = radial_mean_signal(roi_cfi)

                rvals = rm_signal_i[:, 0]
                mean_rs = rm_signal_i[:, 1]
                std_rs = rm_signal_i[:, 2]

                # init the array the first time. We don´t know the rvals.max() before
                if frame_i == 0:
                    rms = np.zeros((rvals.shape[0], nframes))
                    rstd = np.zeros((rvals.shape[0], nframes))
                # assign the radius mean signal values and its std values
                rms[:, frame_i] = mean_rs
                rstd[:, frame_i] = std_rs

            # Store the values if indcated
            if assign:
                roi.rms = rms
                roi.rstd = rstd

            if display:
                # it should be a function in flup, and called here.
                fig, axs = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")

                imi = axs.imshow(rms)

                axs.set_title(f"Colony {col_id}, mean pixel value")

                axs.set_ylabel("radial distance from colony center [px]")
                axs.set_ylim(0, rvals.shape[0])  # decreasing time

                # if xticks were indicated
                if t_axis_pos is not None and t_axis_values is not None:
                    axs.set_xticks(t_axis_pos, t_axis_values.astype(int))
                    axs.set_xlabel("Time [hours]")
                else:
                    axs.set_xlabel("Frame")

                fig.colorbar(
                    imi, fraction=0.035, label="mean pixel value [px]"
                )  # , rotation=270 )

                plt.show()

                # save the image file if indicated
                if ofname is not None:
                    filename = f"{ofname}rms_{chani}_id{col_id}"
                    fig_kwargs = {
                        "bbox_inches": "tight",
                        "transparent": True,
                        "dpi": 300,
                    }
                    save_figure(
                        fig, filename, fformat, overwrite=overwrite, **fig_kwargs
                    )


def im_to_1chan(
    im,
    channels=CHANNELS,
    to_gray=False,
    im_plot=False,
    mode="sum",
    dtype=None,
    gray_conv=[0.2989, 0.5870, 0.1140],
    print_type=False,
    **kwargs,
):
    """
    Convert image to one channel by any of these ways:
    1) sum over the indicated channels
    2) mean value across indicated channels
    3) renormalize the data
    4) convert to grayscale
    Minimal in-place edits for memory efficiency.
    """
    # if no dtype specified, use input image dtype
    if dtype is None:
        dtype = im.dtype.name

    cmap = plt.get_cmap("viridis")

    c_positions = list(channels.keys())
    n_schans = len(c_positions)
    h, w = im.shape[0:2]

    if to_gray:
        gray_conv = [gray_conv[i] for i in c_positions if i < len(gray_conv)]
        # cast dot result to float32 directly
        ocim = np.dot(im[:, :, c_positions], gray_conv).astype(np.float32, copy=False)
        dtype = ocim.dtype.name
        cmap = plt.get_cmap("gray")
        title = "Grayscale image"
    else:
        # use float32 instead of float64
        ocim = np.zeros((h, w), dtype=np.float32)
        cnames = ""
        for i in c_positions:
            cnames += f"{channels[i]},"
            try:
                # accumulation in-place on float32 buffer
                ocim += im[:, :, i]
            except:
                raise Exception(f"image doesn't have channel {i} ({channels[i]})")
        title = f'Sum of "{cnames[:-1]}" channels'

        if mode == "mean":
            # in-place division to avoid temporary
            ocim /= n_schans
            title = f'Mean of "{cnames[:-1]}" channels'
        elif mode == "renormalize":
            ocim = renormalize(ocim, dtype=dtype, print_type=print_type, **kwargs)
            title = f'Renormalized sum of "{cnames[:-1]}" channels'

    if mode != "renormalize":
        ocim = change_dtype(ocim, dtype, print_type=print_type)

    if im_plot:
        plt.figure()
        plt.imshow(ocim, cmap=cmap)
        plt.colorbar()
        plt.title(title)

    return ocim



def change_dtype(array, dtype, print_type=False, forze=False):
    """
    To change the array data type to another one without suffer overflow
    or underflow (in a "safe" way)
    if the objective data type suffers one of these problems, dtype will be
    changed to the next data type that supports them (e.g. if the array has a
    value equal to 300, and dtype is 'uint8', which supports until 255, dtype
    will be changed to 'uint16').
    Also it rounds the integer values instead of flooring them as the default
    narray.astype() function behaviour

    Parameters
    ----------
    array:
        numpy array to be changed the data type

    dtype: data type indicator (string or dtype)
        desired data type to convert the input array.
        e.g.: 'float64', int, 'int8', np.int8, np.uint8, 'uint8', etc

    print_type: bool
        if True, the finally used data type will be printed.

    forze: bool, optional
        if True, the image is forced (using 'clip') to be the indicated dtype,
        losing the values outside data type limits.

    Returns
    -------
    array:
        return the array with its data type equal to the indicated one or
        the nearest (in the sense of memory usage).
    """
    # if dtype is not a string, obtain its name string
    if type(dtype) != str:
        try:
            dtype = dtype.name
        except:
            dtype = np.array([], dtype=dtype).dtype.name

    # auxiliar integers type list
    uints = [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]  # "unsigned" integers (non-negative)
    ints = [
        "int8",
        "int16",
        "int32",
        "int64",
    ]  # symmetric integer range (negative to positive)

    # in case of any class of integer data type (uint8, int32, etc), round the value
    # transforming them directly with dtype, rounding up instead of flooring.
    if np.issubdtype(np.dtype(dtype), np.integer):

        # round the array values (using round_up function)
        array = round_up(
            array
        )  # round up (0.5 to 1) and transform to int64 or int32 (depends on OS)

        # check the data is inside the datatype format limits
        type_lims = np.iinfo(dtype)  # get the input data type limits

        # start from the dtype element position
        try:
            i = uints.index(dtype)
        except:
            try:
                i = ints.index(dtype)
            except:
                raise Exception(
                    f"\n{dtype=} is not in the function integer type lists...\n"
                )

        # perform the check, initializing the new data type variable
        ndtype = dtype

        while True:
            if array.max() > type_lims.max:
                i += 1
                ndtype = uints[i]
                type_lims = np.iinfo(ndtype)
            else:
                break

        # if it has negative values
        while True:
            if array.min() < type_lims.min:
                ndtype = ints[i]
                type_lims = np.iinfo(ndtype)
                i += 1
            elif array.max() > type_lims.max:
                ndtype = ints[i]
                type_lims = np.iinfo(ndtype)
                i += 1
            else:
                break
    else:
        # For non-integer dtypes, simply use the provided dtype.
        ndtype = dtype

    if forze:
        if np.issubdtype(np.dtype(dtype), np.integer):
            array = np.clip(array, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(
                dtype
            )
        else:
            array = array.astype(dtype)
        if ndtype != dtype:
            print(f"Data format was in range {ndtype} but forzed to {dtype}")
    else:
        # ensure the data type is the appropriate one
        array = array.astype(ndtype)

    # display the used dtype
    if print_type:
        if ndtype != dtype:
            print(f"Data format was changed from {dtype} to {ndtype}")
        else:
            print(f"Finally used data type: {ndtype}")

    return array


# Delete this function!
def forze_imdtype(array, dtype, renorm=False, factor=1):
    """
    ** Deprecated unnecesary and unfinished function **
    ** renormalize() works perfectly in any case using the propper parameters **

    renormalize the array to the input dtype.
    dtype coul be just uint8, float32 or float32.
    if renoralize is True, it will renormalize for
    [0,255] for uint8 and to [0,1] for floats.
    if factor is indicated, the data will be multiplied for that
    factor, and then normalized if its indicated.

    Parameters
    ----------
    array:
        numpy array to be changed the data type

    dtype: data type indicator (string or dtype)
        desired data type to convert the input array.
        options: 'uint8' or np.uint8, 'float32' or np.float32,
                'float64' or np.float64

    renorm: bool or list
        if True, the data will be renormalized accord dtype.
        if list, the data will be renomalize accord its indicated values.
        e.g. It has to be a list with two numbers: [vmin, vmax] of
        the normalization.

    factor: numeric
        modify the data by this factor in a way to minimize data loss.

    Returns
    -------
    array:
        return the array with its data type equal to the indicated one
        and renormalized if indicated

    """
    # data type options
    ftypes = ["float16", "float32", "float64", "float128"]
    uitypes = ["uint8"]

    # if dtype is not a string, obtain its name string
    if type(dtype) != str:
        try:
            dtype = dtype.name
        except:
            dtype = np.array([], dtype=dtype).dtype.name

    # get the input array dtype
    idtype = array.dtype.name

    # now we have differente situation depending on the combination
    # of the array format and the objective format of convertion

    # if idtype is 'float
    if np.issubdtype(idtype, np.floating):

        # if the convertion format is uint8
        if dtype == "unit8":

            if renorm or isinstance(renorm, list):
                a = False  ## just to fill this place

        # if the convertion is another float
        elif np.issubdtype(dtype, np.floating):
            array = change_dtype(array, dtype)

        # no other types are allowed
        else:
            raise Exception("\nInvalid objective dtype")

    elif np.issubdtype(idtype, np.integer):
        a = False  ## just to fill this place
    # isinstance(variable, (int, float, complex, np.number)
    # np.iinfo para integers np.finfo para floats
    return array


def channels_sum(rois_data, cv, channels=CHANNELS):
    """
    Compute the sum over the RGB channels for each image

    Parameters
    ----------
    rois_data: dictionary
            RGB time-lapse image data of each ROIS, from obtain_rois()

    cv: vector
            contain the ID of the of colonies analysed

    channels = dict
        dictionary with the channels positions and names.
        e.g. channels = { 0 : 'R', 1 : 'G'}, indicate the image has the channels
        R and G inpositions 0 and 1 respectivelly.

    Returns
    -------
        sum_chan_rois: dictionary
            Sum of channels for each time step and ROI
    """

    c_positions = list(channels.keys())

    sum_chan_rois = {}
    for i in cv:
        sum_chan_rois[i] = np.zeros((rois_data[channels[c_positions[0]]][i].shape))

    for position in c_positions:
        c = channels[position]
        for i in cv:
            sum_chan_rois[i] += rois_data[c][i][:, :, :]

    return sum_chan_rois


def frame_colony_radius(rois, cv, thr, min_sig=0.5, max_sig=10, num_sig=200):
    """
    Get the colony radius at each time step

    Parameters
    ----------
        rois: dictionary
            ROI image data from obtain_rois()

        cv: vector
            contain the ID of the of colonies analysed

        thr: double
            Threshold for skfeat.blob_log

        min_sig: double
            minimum value of sigma used on skfeat.blob_log

        max_sig: double
            maximum value of sigma used on skfeat.blob_log

        num_sig: int
            number of sigma values used between min_sig and max_sig on skfeat.blob_log


    Returns
    -------
        R: dictionary
            The time series of colony radius size, indexed by colony id number.

    """
    R = {}
    nt = rois[cv[0]].shape[2]
    for k in cv:
        R[k] = np.zeros((nt,))
        for i in range(nt):
            troi = rois[k][:, :, i].astype(np.float32)
            if len(troi):
                nt_roi = (troi - troi.min()) / (
                    troi.max() - troi.min()
                )  # Normalization
                AA = skfeat.blob_log(
                    nt_roi / 1,
                    min_sigma=min_sig,
                    max_sigma=max_sig,
                    num_sigma=num_sig,
                    threshold=thr,
                    overlap=0.8,
                )
                # AA = skfeat.blob_log(nt_roi, min_sigma=0.1, max_sigma=6.0, num_sigma=150, threshold=thr, overlap=0.8)
                if len(AA) > 0:
                    R[k][i] = AA[0, 2] * (2)
                    # R[k][i] = AA[0,2]*(2**0.5)
    return R


def area(r, cv, T, filename="null", fformat="pdf", overwrite=False):
    """
    Compute and plot the colonies area over time as a perfect circle (using
    the input radius value) around the colony position value

    Parameters
    ----------
        r: dictionary
            colony radius at each time step of the selected colony (obtained with frame_colony_radius() function)

        cv: vector
            colonies ID vector to plot


        T: vector
            the vector of real time values

        filename: string
            filename to save the plot generated

    Returns
    -------
        A: dictionary
         colony area at each time step of the selected colony. Call it as: A[colonyID][time step]
    """
    plt.figure()
    A = {}
    for i in cv:
        R = r[i]
        A[i] = np.pi * R * R
        plt.plot(T, A[i], ".", label="colony " + str(i))

    if filename != "null":
        fig = plt.gcf()
        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        save_figure(fig, filename, fformat, overwrite=overwrite, **fig_kwargs)

    return A


def f_sigma(t, a, b, c):
    """
    Compute the sigmoide function value using the given input values

    Parameters
    ----------
        t: vector
            independent variable ( "x axis", suposed to be time)

        a: double
            maximum value parameter

        b: double
            function parameter

        c: double
            delay parameter

    Returns
    -------
    function evaluation

    """
    return a / (1 + np.exp(-(t + b) * c))
    # return((a /(1+np.exp(-(t+b)*c)))+d)


def function_fit(
    xdata,
    ydata,
    init,
    end,
    cv,
    func=f_sigma,
    param_bounds=([1, -np.inf, 0.1], [np.inf, -1, 1]),
):
    """
    Fit a given function to given data

    Parameters
    ----------
        xdata: vector
            independent variable ( "x axis", suposed to be time vector)

        ydict: array like
            array of dependent variable vectors

        init: double
            point on the time vector to init the fitting

        end: double
            point on the time vector to end the fitting

        cv: vector
            contain the ID of the colonies to analyse

        func: function
            function to be fitted

        param_bounds: array of vectors
            lower and upper bounds of each parameters
            para_bounds=([lower bounds],[upper bounds])

    Returns
    -------
        Y_fit: dictionay
            contain the fitting result for each colony in the dictionary.
            It is:

            Y_fit[col ID][evalF z]:

                evalF: vector
                    result vector of the fitted function:
                    evalF=func(xdata, optimal_parameters)

                z: vector
                    fitted parameters

    """

    Y_fit = {}
    for i in cv:
        z, _ = curve_fit(func, xdata[init:end], ydata[i][init:end], bounds=param_bounds)
        print(z)
        evalF = func(xdata, z[0], z[1], z[2])
        plt.plot(xdata, ydata[i], ".", xdata, evalF, "-")
        plt.title("Colony " + str(i))
        plt.show()
        Y_fit[i] = evalF, z
    return Y_fit


def croi_mean_int_frames(data, blobs, radii, cv, channels=CHANNELS):
    """
    compute the mean intensity values for each time and channels for each CROI
    (circular ROI), redefining the ROIS based on radii values
    It takes the fit radius value at each time (radii), with it defines a
    circular ROI, sum all the pixel values inside them and divide this value
    for the number of pixel considered. --> obtain the intensity mean value
    inside the colony limits on each time.

    Parameters
    ----------
        data: dictionary
             RGB dictionary with the images data

        blobs: array like
            contains the information of identified blobs

        radii: dictionary
            contains the radius for each colony on each time step

        cv: vector
            contain the ID of the colonies to analyse

        channels: dict
            dictionary with the channels positions and names.
            e.g. channels = { 0 : 'R', 1 : 'G'}, indicate the image has the channels
            R and G inpositions 0 and 1 respectivelly.
    Returns
    -------
        all_chan_crois_mean_val: dictionary
            contain the mean pixel value of each channel for each time step of each colony.
            call it as: all_chan_crois_mean_val['channel_name'][blob_number][timepoint]


    """
    all_chan_crois_mean_val = {}

    for key in channels.keys():
        chan = channels[key]
        crois_mean_val = {}

        for i in cv:
            # x and y are the colony center pixel stored on blobs
            x = blobs[i, 0]
            y = blobs[i, 1]
            meanInt = np.zeros((len(radii[i])))

            for j in range(len(radii[i])):
                ####### this lines are to eliminate the out of image bounds error
                r = radii[i][j]

                # Define bounds to avoid out-of-image errors
                x1 = max(round_up(x - r), 0)
                x2 = min(round_up(x + r + 1), data[chan].shape[0])
                y1 = max(round_up(y - r), 0)
                y2 = min(round_up(y + r + 1), data[chan].shape[1])
                #######
                SRoi = data[chan][x1:x2, y1:y2, j]

                # Define the center of the subregion
                xc = (SRoi.shape[0] - 1) / 2  # int((SRoi.shape[0]+1)/2)
                yc = (SRoi.shape[1] - 1) / 2  # int((SRoi.shape[1]+1)/2)

                ##

                # Create a grid for the subregion
                xx, yy = np.ogrid[: SRoi.shape[0], : SRoi.shape[1]]

                # Create the circular mask
                mask = (xx - xc) ** 2 + (yy - yc) ** 2 <= r**2

                # Compute the mean intensity for the masked region
                if np.any(mask):
                    meanInt[j] = SRoi[mask].mean()

                ##
                # for n in range(SRoi.shape[0]):
                #    for m in range(SRoi.shape[1]):
                #        if ((n-xr)**2+(m-yr)**2) <= (r**2):
                #            CRoi_int += SRoi[n,m]
                #            count += 1
                # if count != 0:
                #    meanInt[j] = CRoi_int/count

            crois_mean_val[i] = meanInt
        all_chan_crois_mean_val[chan] = crois_mean_val

    return all_chan_crois_mean_val


def f_mu(t, b, d):
    """
    compute the grwoth rate (mu) function value

    Parameters
    ----------
        t: int or vector
             independent variable values (suposed to be time vector)

        b: double
            functon parameter

        c: double
           function parameter


    Returns
    -------
        evaluated "mu" fucntion with the given parameters


    """
    return d / (np.exp(d * (t + b)) + 1)


def f_linear(x, a, b):
    """
    compute the linear function value with given parameters

    Parameters
    ----------
        x: int or vector
            independent variable values

        a: double
            slope parameter

        b: double
           y-intercept parameter


    Returns
    -------
        evaluated linear fucntion with the given parameters for the given x
    """

    return a * x + b


def quadratic_2d(X, a, b, c, d, e, f):
    x, y = X
    return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f


def paraboloid_2d(X, a, b, c, d, e):
    x, y = X
    return a * (x - d) ** 2 + b * (y - e) ** 2 + c


def linear_fit(data1, data2, filename="null", fformat="pdf", overwrite=False):
    """
    Fit linear function (f_linear) to given data, display the fited function
    and make a plot of the result. You are able to save the resulting plot by
    given as input the "filename" to save it.

    Parameters
    ----------
        data1: vector
            independent variable ( "x axis") to be used as input of f_linear

        data2: vector
            "y-data values" used as reference to peform the fitting

        filename: string
            name of the image file if it is desired to save it.


    Returns
    -------
        z: vector
            fitted parameters

    """

    z, _ = curve_fit(f_linear, data1, data2, bounds=([0, -np.inf], np.inf))
    # print(z)           #first component is the slope
    p = np.poly1d(z)
    print(np.poly1d(p))
    xp = np.linspace(data1.min(), data1.max(), 2)
    # plt.plot(timeC[init:end], ratio[init:end,i], '.', xp, p(xp), '-')
    plt.figure()
    axisMax = np.max([np.max(data1), np.max(data2)])
    axisMin = np.min([np.min(data1), np.min(data2)])
    plt.axis([axisMin, axisMax, axisMin, axisMax])
    plt.plot(data1, data2, ".", xp, p(xp), "-")

    if filename != "null":
        fig = plt.gcf()
        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        save_figure(fig, filename, fformat, overwrite=overwrite, **fig_kwargs)

    plt.show()

    return z


def colony_classifier(fit, classes, chanx_dat, chany_dat):
    """
    Classify chanx_dat and chany_dat (which correspond to the data serie being
    classified) on the classes names given as inputs. The classification is
    on the small chany_dat distance value with the value computed with the
    chanx_dat and each fitted function (that are on "fit"). In other words,
    accord the minimal y-coordinate distance between each dot and fitted lines.

    Parameters
    ----------
        fit: array like
            each position on the array cointains the parameters of the linear
            fit of each categorie. fit = [z1, z2, z3] where z is the return of
            linear_fit.

        classes: string array
            contain the names of the defined categories (its length have to be
            same long as fit)

        chanx_dat: vector
            data of the channel on x axis for the data to be classified

        chany_dat: vector
            data of the channel on y axis for the data to be classified


    Returns
    ----------
        clas: list
            contain the category of each classified colony in order.
            e.g. clas = ['cat3', 'cat1, 'cat1', 'cat1', 'cat2', etc ...]

        clas_dict: dictionary
            contain the channel value of the colonies of each category in the
            corresponding dictinary class. clas_dict = ['class'][chan_xdat,
            chany_dat, boolean]. The boolean vector have the length of the
            total colony analyzed, and indicate (with True) which colonies
            correspond to that category.

    """
    CAT_NUM = len(fit)  # number of categories
    y = np.zeros(CAT_NUM)
    d = np.zeros(CAT_NUM)
    clas = np.zeros(len(chanx_dat))
    clas_dict = {}

    # evaluate if have same number of classes as linear fits
    if CAT_NUM == len(classes):

        # compute the difference between the straight lines categories and the
        # colonies being classified
        for i in range(len(chanx_dat)):
            for j in range(CAT_NUM):
                y[j] = fit[j][0] * chanx_dat[i] + fit[j][1]
                d[j] = (y[j] - chany_dat[i]) * (y[j] - chany_dat[i])

            # find the minimal difference value
            mindif = np.min(d)

            # perform the classification
            TOKEN = 0
            count = 0
            while TOKEN == 0:
                if mindif == d[count]:
                    clas[i] = count
                    TOKEN = 1
                count += 1

        # store the data in a dictionary of categories
        for n in range(len(classes)):
            clas_dict[classes[n]] = [
                chanx_dat[clas == n],
                chany_dat[clas == n],
                clas[:] == n,
            ]

        # save a list with the corresponding string category name of each element in clas
        roi_clas = []
        for i in range(len(clas)):
            roi_clas.append(classes[int(clas[i])])

        return (roi_clas, clas_dict)

    else:
        print("\nERROR: classes have to be same length as fits\n")


# End
