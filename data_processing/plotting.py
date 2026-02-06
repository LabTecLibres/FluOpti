# -*- coding: utf-8 -*-
"""
Plotting functions developed for the FluoPi/FluOpti proyect.

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
import inspect  # to manage **kwargs

from scipy.interpolate import UnivariateSpline

# to use interactive widgets
import ipywidgets as widgets
from IPython.display import display, clear_output

# import fluopi functions
try:
    import analysis as flua
except:
    from fluopi import analysis as flua

###################
## General lists ##

# Define the image default channels dimentional position and name
CHANNELS = {0: "R", 1: "G", 2: "B"}
# structured as {channel_position : channel_name}
# --> channels_positions = list(CHANNELS.keys())

# default color to be used in plots for each channel position
COLORS = {0: "r", 1: "g", 2: "b"}

###########################
####### Functions #########


def get_colors(cmap_name, n=None, ids=None, dkeys=None, get_cmap=False):
    """
    To obtain a list or dictionary of equidistant sampled colors from a colormap list
    (i.e. interpolated values from the indicated colormap). If dkeys is indicated,
    a dictionary with that keys will be returned, otherwise it will be a list.

    Optinally you can indicate specific ids to pick those colors from the
    colormap (id values between [0,n]). It is useful if you want to use just
    half of a colormap. e.g. n = 10, ids = [0,1,2,3,4].


    Parameters
    ----------
    cmap_name: string
        colormap name.
        e.g. 'viridis'

    n: int
        number of colors to create.
        If ids are not indicated, all of them will be returned

    ids: list of integers
        defined indices to be used of the n colors generated.
        they have to be integer values between [0,n]
    dkeys: list
        if indicated, a dictionary with the colors will be returned asociated
        to this dkeys

    get_cmap: bool
        if True, an asociated colormap object is returned.

    Return
    ------
    colors: list or dictionary
        list or dict with n (or length ids) colors from the choosen colormap
        accord the indicated indices or lineally distributed in the colormap space.

    ncmap: [optinal]
        you can also return the cmap
    """

    # get the indicated colormap
    cmap = mpl.colormaps.get_cmap(cmap_name)

    # define the number of colors
    if n is None:

        # create as much as ids where indicated
        if ids is not None:
            n = len(ids)

        # othewise create one for each dictionary
        elif dkeys is not None:
            n = len(dkeys)

        else:
            raise Exception(
                "You have to indicate at least one of 'n', 'ids' or 'dkeys' parameters"
            )

    # with it, create the n colors
    cmap = cmap.resampled(n)

    # in case ids was not indicated, get the lineally distributed indices
    if ids is None:

        ids = np.arange(n)

    # pick those color positions by interpolation of each channel
    # colors = np.array([np.interp(ids, np.arange(N), cmap_colors[:, i]) for i in range(cmap_colors.shape[1])]).T
    colors = cmap(ids)

    # create an output dictionary if dkeys was indicated.
    if dkeys is not None:

        d_colors = dict()

        for i in range(len(dkeys)):

            # asociate the color of the "i" id.
            d_colors[dkeys[i]] = colors[i]

        colors = d_colors

    # optionally you could create a colormap
    if get_cmap:
        ncmap = mpl.colors.ListedColormap(colors)
        return (colors, ncmap)

    # otherwise, return just the colors
    return colors


def lighten_color(color, amount=0.5):
    """
    lighten an input color given in Matplotlib format (hexadecimal or name).

    Parameters:
    color: matplotlib color
        input color to be ligthen.
    amount: numeric
        amount of lightening indicated by a value between [0,1]
        (0 = without change, 1 = white).
    """
    try:
        c = mpl.colors.cnames[color]
    except KeyError:
        c = color

    if amount > 1 or amount < 0:
        print("\nInvalid lighting amount value. Default 0.5 will be used.")
        amount = 0.5

    c = mpl.colors.to_rgb(c)

    return tuple(1 - (1 - component) * amount for component in c)


def im_hist(
    im_vectors,
    groups="unitary",
    colors=COLORS,
    density=False,
    labels=None,
    line_colors=["k"],
    **kwargs,
):
    """
    It creates an histogram of the pixel values for each image channel.
    If more than one image is given, the first histogram it represented by solid bars
    and the next ones by solid step lines.

    In case the given image channels are not converted to a vector, this function
    convert them with flatten() method.


    Parameters
    ----------
    im_vectors : list of dictionaries or 1d-arrays
        vectors of each image channel (the image should be converted to a vector
        otherwise this function automtivally convert it to 1D with flatten()))
        it could be a list with two elements, in that case plot the first one
        as solid bars and the second one as a line.

    groups : int or sequence
        by default its value is "unitary", in this case the number of bins is
        computed for each im_vector to be width = 1.
        if int, is the number of equal size histograms bins/groups/bars
        if sequence (list or array), "it defines the bin edges, including the
        left edge of the first bin and the right edge of the last bin;
        in this case, bins may be unequally spaced."

    colors : dict or color
        dict with the colors to be used in the plot of each channel position
        e.g. {0:'r', 2: 'b'}, will display a red line for channel 0 and a
        blue line for channel 2.
        if image(s) has just one channelthe the first color of colors is used
        at least another valid input is given (e.g. colors = 'r')

    density : boolean
        "If True, draw and return a probability density:
        each bin will display the bin's raw count divided by the total number
        of counts and the bin width
        (density = counts / (sum(counts) * np.diff(bins))),
        so that the area under the histogram integrates to 1
        (np.sum(density * np.diff(bins)) == 1)."

    labels: list of strings
        String with the labels associated to each input image (in the same order)
        If no label is indicated, it is filled with the input order of the images.

    line_colors: list of color elements
        color used to display the line of the "step type" histogram(s)

    **kwargs: keyword arguments
        Any other(s) arguments to pass to .hist()
        like "alpha" or "lw"

    Returns
    --------
    outputs : list
        list with the histogram outputs.
        i.e. the counts and bins of each one.
        e.g. outputs[0] contains the tuple (counts, bins) of the first histogram

    """
    # in case type is not a list, convert to list for code consistency
    if type(im_vectors) != list:
        im_vectors = [im_vectors]

    # init the output list
    outputs = list()

    # go for each image
    for i, imv in enumerate(im_vectors):

        # get the label of the image
        if type(labels) == list:
            try:
                label_i = labels[i]
            except:
                label_i = f"image #{i+1}"
        else:
            label_i = f"image #{i+1}"

        ### in case the image data is in a dictionary
        if type(imv) == dict:

            output_i = dict()

            if i == 0:
                chans = list(imv.keys())
                nchans = len(chans)

                # plt.figure(figsize=(15,3))
                fig, axs = plt.subplots(
                    nrows=1, ncols=nchans, figsize=(4 * nchans, 3), layout="constrained"
                )

                for c in chans:

                    # make sure the image channel is a vector
                    imv[c] = imv[c].flatten()

                    # define the bins for the histogram
                    if groups == "unitary":
                        bins = imv[c].max() - imv[c].min()
                    else:
                        bins = groups

                    # make an histogram for each image channel
                    output_i[c] = axs[c].hist(
                        imv[c],
                        bins,
                        density=density,
                        facecolor=colors[c],
                        label=label_im,
                        **kwargs,
                    )  # , alpha=0.75, histtype='stepfilled'
                    axs[c].set_xlabel("pixel value")

                    if c == 0:
                        if density == False:
                            axs[c].set_ylabel("pixel count")
                        else:
                            axs[c].set_ylabel("pixel prob density")

                    try:
                        axs[c].set_title("Channel " + str(chans[c]))

                    except:
                        axs[c].set_title("Channel " + str(c))

                outputs.append(output_i)

            else:
                for c in chans:
                    # make sure the image channel is a vector
                    imv[c] = imv[c].flatten()

                    # get the idicated line color
                    line_color = line_colors[i - 1]

                    # define the bins for the histogram
                    if groups == "unitary":
                        bins = imv[c].max() - imv[c].min()
                    else:
                        bins = groups

                    # add the values to the corresponding ax
                    output_i[c] = axs[c].hist(
                        imv[c],
                        bins,
                        density=density,
                        color=line_color,
                        histtype="step",
                        label=label_i,
                        **kwargs,
                    )

                outputs.append(output_i[c])

            # add the last legend
            axs[c].legend()

        # in case it is not a dictionary (i.e. monochromatic image)
        else:

            # make sure the image channel is a vector
            imv = imv.flatten()

            # get the color to make the plot
            if type(colors) == dict:
                # get the fist color
                color = colors[list(colors.keys())[0]]
            else:
                color = colors

            # create the figure for the fist hist
            if i == 0:
                fig, ax = plt.subplots(figsize=(8, 6))

                # define the bins for the histogram
                if groups == "unitary":
                    bins = imv.max() - imv.min()
                else:
                    bins = groups

                # create the histogram
                output_i = ax.hist(imv, bins, facecolor=color, label=label_i, **kwargs)

                ax.set_xlabel("pixel value")
                ax.set_ylabel("pixel count")
                ax.set_title("Unique channel")

            # add the next one(s)
            else:

                # get the idicated line color
                line_color = line_colors[i - 1]

                # define the bins for the histogram
                if groups == "unitary":
                    bins = imv.max() - imv.min()
                else:
                    bins = groups

                # add the step type histogram (i.e. the line) over the bars
                output_i = ax.hist(
                    imv,
                    bins,
                    density=density,
                    color=line_color,
                    histtype="step",
                    label=label_i,
                    **kwargs,
                )

            ax.legend()

            # append the histogram output
            outputs.append(output_i)

    plt.show()

    return outputs


def plot_im_frame(
    fnames, frame, get_im=False, display_info=False, display_im=True, fn_key="fnames"
):
    """
    To plot or get an image frame

    Parameters
    ----------
    fnames : string, list or dictinary

        string: filenames string pattern of images. e.g: my_folder/image%003d.png
        list: with the images filenames
        dict: with a list of fnames insides the fn_key. e.g: fnames[fn_key] = list_filenames

    frame : int
        frame number to plot

    get_im: boolean
        True to return the image array

    display_info: boolean
        True to display some imagen information (e.g. shape)

    display_im: boolean
        True to display the image

    Returns
    ----------
    im : array like
        optional return. Image data

    """

    # get the image path
    if type(fnames) == str:
        im_fpath = fnames % frame

    elif type(fnames) == list:

        if frame == -1:
            frame = len(fnames) - 1

        im_fpath = fnames[frame]

    elif type(fnames) == dict:

        fnames = fnames[fn_key]

        if frame == -1:
            frame = len(fnames) - 1

        im_fpath = fnames[frame]

    else:
        raise Exception("\nInvalid filenames input")

    # get the image data
    im = cv2.imread(im_fpath, cv2.IMREAD_UNCHANGED)

    # convert to RGB if it is multichromatic
    try:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    except:
        print("\nMonocromatic image\n")

    # get image name
    im_fname = im_fpath.split("/")[-1]

    if display_info:

        print("Image shape: " f"{im.shape}\n")
        # f'heigh = {im.shape[0]}, wide = {im.shape[1]}, channels = {im.shape[2]}\n')

    # display it
    if display_im == True:
        plt.figure()
        plt.imshow(im)
        plt.title("frame " + str(frame) + " (" + im_fname + ")")

    if get_im:
        return (im, im_fname)


def plot_data_frame(
    dframe,
    dfname="",
    cmap="viridis",
    display_info=False,
    ax_off=False,
    ofname=None,
    **kwargs,
):
    """
    To plot a data frame

    Parameters
    ----------
    dframe : ndarray
        data frame ndarray

    dfname: str or numeric
        data frame name to be used in the title

    display_info: boolean
        True to display some imagen information (e.g. shape, )

    ofname: string
        if given, the figure is stored under that name.
        More options with **kwargs  (see flua.save_fig())

    Returns
    ----------
    im : array like
        optional return. Image data

    """

    if display_info:

        print(f"\nData shape: {dframe.shape}")
        print(f"Data type: {dframe.dtype}")
        print(f"Data limits: [{dframe.min()},{dframe.max()}]\n")
        # f'heigh = {im.shape[0]}, wide = {im.shape[1]}, channels = {im.shape[2]}\n')

    # display it
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
    # fig, ax = plt.subplots(figsize=(10, 6))

    if dframe.ndim > 2:
        # multichannels image
        ax.imshow(dframe)

    else:
        # mochromatic image
        ax.imshow(dframe, cmap=cmap)

    if ax_off:
        ax.set_axis_off()

    ax.set_title(f"frame {dfname}")

    # save the figure image file if indicated
    if ofname is not None:

        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        fig_kwargs.update(kwargs)  # join with the input functions kwargs
        flua.save_figure(fig, ofname, **fig_kwargs)


def plot_frame_surface(
    im,
    cmap="viridis",
    downscale=1,
    get_im_array=False,
    elev=30,
    azim=45,
    ofname=None,
    **kwargs,
):
    """
    Perform a surface 3d visualization of a monochromatic image.
    --> añadir guardar la figura

    Parameters
    ----------
    im : np.array or dict
        mochromatic image data.

    cmap : str, optional
        colormap to be used (default 'viridis').

    downscale : int, optional
        factor used to downscale the image in order to increase the performace
        speed of displaying. It is very useful with large images.
        e.g. downscale = 2 reduce the resolution (dimentions size) by half.

    elev : int, optional
        Elevation angle of the 3D view.

    azim : int, optional
        Rotation angle of the 3D view.

    get_im_array: bool, optional
        if True, the displayed image data is returned (useful to verify the
        values after rescaling)

    ofname: string
        if given, the figure is stored under that name.
        More options with **kwargs  (see flua.save_fig())

    Return
    ------
        im: np.array, optional
            Finally displayed image after the rescaling
            just returned if get_im_array is True

    """

    # store the input image data type
    imdtype = im.dtype

    # apply the downscale if given
    if downscale > 1:
        im = cv2.resize(
            im,
            (im.shape[1] // downscale, im.shape[0] // downscale),
            interpolation=cv2.INTER_AREA,
        )

        # make sure the image data type is keep the same as input
        if im.dtype != imdtype:
            im = flua.change_dtype(im, dtype, print_type=True, forze=True)

    # get the dimentions
    h, w = im.shape

    # create the X, Y meshgrid of the image size
    x = np.arange(0, w)
    y = np.arange(0, h)
    x, y = np.meshgrid(x, y)

    # create the 3D figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection="3d")  # indicate 3D proyection

    # set the view angle
    ax.view_init(elev, azim)

    # plot the surface
    surf = ax.plot_surface(x, y, im, cmap=cmap, edgecolor="none")

    ax.set_xlabel("X [pixels]")
    ax.set_ylabel("Y [pixels]")
    ax.set_zlabel("Intensity")
    ax.set_title("Image surface")

    # add the colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Intensity")

    # display
    plt.show()

    # save the figure image file if indicated
    if ofname is not None:

        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        fig_kwargs.update(kwargs)  # join with the input functions kwargs
        flua.save_figure(fig, ofname, **fig_kwargs)

    if get_im_array:
        return im


def plot_frame_channels(im=None, f_path="", frame=0, channels=CHANNELS, cmap="viridis"):
    """
    To perform separated plots of each channel of an image frame.
    Image frame could be given directly as input or read from f_path and frame.

    Parameters
    ----------
    im : np.array or dict
        image data.
        If array, the channels are in the 3rd dimention and frame is not used.
        if dict, the channels are its keys. im[channel][n,m,frame]

    f_path : string_pattern or list
        filename path pattern (e.g. folder/image%3d)
        or explicit filename path list

    frame : int
        Frame number to plot

    channels: dict
        Channels position and name information.
        Structured as {channel_position : channel_name}

    cmap: colormap
        colormap to be used in the plots
        #matplotlib.colormaps.get_cmap('viridis')
    """
    # get the channels positions
    c_positions = list(channels.keys())
    nchans = len(c_positions)

    ftext = ""
    # read the image from f_path if no image were input
    if im is None:

        ftext = f" ({frame=})"

        try:
            im_fpath = f_path % frame
        except:
            im_fpath = f_path[frame]

        im = cv2.imread(im_fpath, cv2.IMREAD_UNCHANGED)

        # in case of multicromatic image, convert to RGB
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    try:
        nchans = im.shape[2]
    except:
        pass

    if nchans > 1:

        fig, axs = plt.subplots(
            nrows=1, ncols=nchans, figsize=(4 * nchans, 3), layout="constrained"
        )

        for i in c_positions:

            if type(im) == dict:
                imi = im[channels[i]][:, :, frame]
                ftext = f" ({frame=})"
            else:
                imi = im[:, :, i]

            cax = axs[i].imshow(imi, cmap=cmap)
            axs[i].set_title(f"{channels[i]} channel")
            fig.colorbar(cax, fraction=0.035)
            fig.suptitle("Image" + ftext, fontsize=12)
    else:

        title_text = "Monochromatic image"

        if type(im) == dict:
            chan = channels[c_positions[0]]
            im = im[chan][:, :, frame]
            title_text = f"'{chan}' channel"
            ftext = f" ({frame=})"

        plt.figure()
        plt.imshow(im, cmap=cmap)
        plt.colorbar()
        plt.title(title_text + ftext)


def row_transect(
    data,
    rows,
    data_frames=-1,
    colors="jet",
    channels=CHANNELS,
    show_im=True,
    cmap="viridis",
    max_ims=3,
    color_ids=None,
    im_labels=None,
    cmap_range=None,
):
    """
    Plot the value of a transect(s) (row of pixels) in a frame and plot it

    Parameters
    ----------
    data : dictionary or numpy array
        dictionary with the R G B data of all images
        estructure: data[channel][rows,columns,frame_number]
        in case of input a narray, [rows, columns, channels]

    rows : int or list of ints
        row(s) where you want to see the transect(s)

    data_frames: int or list
        frame number(s) of the image(s) of interest, default = last one

    colors: list, numpy array or colormap name
        colors to be used in the lines of each data_frame (i.e. input
        one for each data_frame)
        if input a colormap, the colors will be assigned uniformly from it

    show_im: boolean or list
        if True, the image(s) is/are displayed
        if data_frames are more than 3, instead of boolean
        indicate the frames numbers to be displayed.

    channels: dict
        dictionary with the channels positions and names
        e.g. channels = { 0 : 'R', 1 : 'G'}, indicate the image has the channels
        R and G inpositions 0 and 1 respectivelly.
        In this case this information is used to recompose the image

    cmap: colormap
      [optional] matplotlib colormap name to be used in the image composition display.

    max_ims: int
        maximum number of images to display automatically when show_im is just
        a boolean. (it avoids displayeing the images too small)

    color_ids: None
        list with specifics ids of the colormap

    im_labels: list
        list with the titles/labels used for each indicated image

    cmap_range: list with two int
        list with [vmin, vmax], where:
        vmin = colormap minimum value
        vmax = colormap maximum value
        If no specified, it takes the images maximum and minimum
    """
    # ids of the images to display
    im_ids = None

    # check data type and get the number of data frames
    if type(data_frames) == int:

        data_frames = [data_frames]

    elif type(data_frames) == list or type(data_frames) == np.ndarray:
        pass

    else:
        raise Exception("data_frames has to be an int or a list")

    nframes = len(data_frames)

    #########################
    ### Transects display ###
    #########################

    # get the dataframes colors in case indicated a colormap name
    if type(colors) == str:

        colors = get_colors(colors, n=nframes, ids=color_ids)

    # check rows type
    if type(rows) == int:

        rows = [rows]

    # get the position and number of channels
    c_positions = list(channels.keys())
    nchans = len(c_positions)

    # in case data is directly an image frame
    if type(data) == np.ndarray:
        imdata = data

        # in case of monochromatic data
        if imdata.ndim == 2:

            # add a channel of length 1 to be consistent with the resto of the code
            imdata = np.expand_dims(imdata, axis=-1)

            # if input channels number is bigger than one channel
            if nchans > 1:

                channels = {0: "Unique"}
                c_positions = list(channels.keys())
                nchans = len(c_positions)

        # create an axis for each channel
        fig, axs = plt.subplots(
            nrows=1, ncols=nchans, figsize=(4 * nchans, 3), layout="constrained"
        )

        # in case of just one channel convert to list for code coherence
        if nchans == 1:
            axs = [axs]

        # display the transects for each channel
        for i in range(nchans):
            pos = c_positions[i]
            chani = channels[pos]

            if i == 0:
                axs[i].set_ylabel("pixel value")

            # make a line for each row
            lines = list()
            try:

                for j in range(len(rows)):
                    row = int(rows[j])  # make sure it is an integer
                    (li,) = axs[i].plot(imdata[row, :, pos], label=f"row {row}")
                    lines.append(li)

                axs[i].set_xlabel("horizontal pixel position")
                axs[i].set_title(f"{chani} channel")
                axs[i].legend(handles=lines)

            except:
                print(f"Channel {chani} cannot be found in the data.")

        # variable to be used in case of display the image
        images = [imdata]
        im_ids = [" "]

    # in case data is a dictionary
    elif type(data) == dict:

        fig, axs = plt.subplots(
            nrows=1, ncols=nchans, figsize=(4 * nchans, 3), layout="constrained"
        )

        # in case monochromatic convert to list for code coherence
        if nchans == 1:
            axs = [axs]

        # go for each indicated channel
        for i in range(nchans):
            pos = c_positions[i]
            chani = channels[pos]

            # add the y label just in the first axis
            if i == 0:
                axs[i].set_ylabel("pixel value")

            # make a line for each indicated row and each indicated image
            lines = list()

            for j in range(len(rows)):
                row = int(rows[j])  # make sure it is an integer
                # and for each frame
                for k in range(nframes):

                    dfk = data_frames[k]

                    # in case -1 was indicated, replace -1 by the image number of the last image
                    if dfk == -1:
                        dfk = data[chani].shape[-1] - 1

                    (lik,) = axs[i].plot(
                        data[chani][row, :, dfk],
                        color=colors[k],
                        label=f"row {row}, frame {dfk}",
                    )
                    lines.append(lik)

            axs[i].set_xlabel("horizontal pixel position")
            axs[i].legend(handles=lines)
            axs[i].set_title(f"{chani} channel")

    ###################################
    # Display the images if indicated #
    ###################################

    # if a list of images was indicated to display
    if type(show_im) == list:
        im_ids = show_im
        show_im = True  # change the display image flag to True

    if show_im:

        # Get the image frame data in case data is a dictionary of channels with image frames
        if type(data) == dict:
            images = list()

            # define the im_ids to display
            if im_ids is None:

                if nframes <= max_ims:
                    im_ids = data_frames

                else:
                    # get the first, the middle and the final ids
                    im_ids = list()
                    for i in np.linspace(0, nframes - 1, max_ims).round().astype(int):

                        im_ids.append(data_frames[i])

                    print(
                        f"As no specific images were indicated to display, \
                    just {max_ims} images will be displayed"
                    )

            # compose the images
            # go for each frame
            template = data[channels[c_positions[0]]]
            n, m = template.shape[0:2]  # take just the first two dimentions

            for i, k in enumerate(im_ids):
                # in cas im_id = -1, replace by the image number of the last image
                if k == -1:
                    im_ids[i] = template.shape[-1] - 1
                    k = im_ids[i]

                # init the image array
                dtype = template[:, :, k].dtype
                imdata = np.zeros((n, m, nchans), dtype=dtype)

                # go for each channel
                for pos in c_positions:

                    chani = channels[pos]
                    imdata[:, :, pos] = data[chani][:, :, k]

                # append the image
                images.append(imdata)

        # Display the image(s)
        nims = len(images)
        fig, axs = plt.subplots(nrows=1, ncols=nims, figsize=(4 * nims, 3))

        # in case of just one image convert to list for code coherence
        if nims == 1:
            axs = [axs]

        # set the limits of the colorbar to min and max values of all the images

        if cmap_range == None:
            cmap_range = [None, None]

            for imi in images:

                # update the values just if it's monochromatic
                if imi.ndim == 2 or imi.shape[-1] == 1:
                    imi_min = imi.min()
                    imi_max = imi.max()

                    if cmap_range[0] == None or cmap_range[0] > imi_min:
                        cmap_range[0] = imi_min

                    if cmap_range[1] == None or cmap_range[1] < imi_max:
                        cmap_range[1] = imi_max

        # go for each image
        for i in range(nims):

            imi = images[i]
            """
            # correct the image format if necessary
            if imi.ndim == 3:
                if imi.dtype != np.float64 and imi.dtype != np.float32:   
                  
                    if imi.max() > 1:
                        imi = imi.round().astype('uint8')   #change the data type to show the image properly
            """

            axs[i].set_title(f"Image {im_ids[i]}")

            # display the image
            if imi.ndim == 2 or imi.shape[-1] == 1:
                axi = axs[i].imshow(
                    imi, cmap=cmap, vmin=cmap_range[0], vmax=cmap_range[1]
                )

                # add the colorbar in the last image
                if i == nims - 1:
                    fig.colorbar(axi, ax=axs[i], fraction=0.04, orientation="vertical")

            else:
                axs[i].imshow(imi)

            # display the transects

            width_i = imi.shape[1]

            for j in range(len(rows)):
                row = int(rows[j])
                rect = Rectangle(
                    (0, row), width_i, 0, linewidth=1, edgecolor="r", facecolor="none"
                )
                axs[i].add_patch(rect)


# añadir para guardar la figura


def circle_pixel_to_mm(ims, d_mm=None, r_px=None, center=None, subdir=""):
    """
    Función interactiva para establecer la relación mm/px
    en base a las imágenes que se ingresan.
    Esto se logra mediante el posicionamiento de un círculo de tamaño conocido.

    Parameters
    ----------
    ims : list
        list of np.ndarray images

    d_mm: numeric
        initial diammeter value (in millimeters)

    r_px : int, optional
        Radio inicial del círculo, in pixels units

    center : list, optional
        Coordenadas iniciales del centro del círculo [y, x].

    subdir: string
        Subdirectoy to save the generated figure

    Returns
    -------
    circle_values: dict
        chosen circle values in pixel units
        ['center'] = [cy, cx]  # circle center coordinates
        ['r'] = radius         # circle radius

    relation: dictionary
        the value of mm to pixel relation

    """

    # global global_fig  # Almacena la figura global para guardar posteriormente
    # init the result dictionary

    # use the first image to get the shape
    im_shape = ims[0].shape
    nims = len(ims)  # number of images

    # get the initial values for center and radius
    if type(center) != list:

        center = [im_shape[0] // 2, im_shape[1] // 2]  # circle center [y,x]

    if r_px == None:

        r_px = im_shape[0] // 2.5  # circle radius (in pixels)

    if d_mm == None:

        d_mm = 1  # circle physical diammeter

    # init the output result
    circle_values = {"center": center, "r": r_px}

    relation = {"mm_px": 1}  # init the dictionary with the mm to pixel relation

    # init the figures and plot the images (just once and they keep)
    # with output:
    # clear_output(wait=True) # Limpiar salida anterior

    # Crear figura y ejes para las dos imágenes
    fig, axs = plt.subplots(1, nims, figsize=(8 * nims, 6))

    # in case of just one image convert to list for code consistency
    try:
        axs[0]
    except:
        axs = [axs]

    # init the figure elements dictionaries
    circles = {}
    lines = {}
    antpx = {}  # pixel annotations
    antmm = {}  # mm annotations

    # annotation coordinates
    ycoords = [center[0], center[0]]  # Coordenadas Y de la línea
    xcoords = [center[1] - r_px, center[1] + r_px]  # Coordenadas X de la línea

    for i, im in enumerate(ims):
        # display the image
        axs[i].imshow(im)  # , cmap='gray')

        # Dibujar los círculos iniciales
        circles[i] = Circle(
            (center[1], center[0]),
            r_px,
            color="r",
            fill=False,
            lw=1,
            label=f"r = {r_px} px",
        )
        axs[i].add_artist(circles[i])

        # Dibujar la línea horizontal en el diámetro inicial
        (lines[i],) = axs[i].plot(xcoords, ycoords, "r")

        # Agregar anotaciones iniciales
        antpx[i] = axs[i].annotate(
            f"{2*r_px:0.1f} px",
            (center[1], center[0] - 30),
            ha="center",
            va="bottom",
            color="red",
        )
        antmm[i] = axs[i].annotate(
            f"{d_mm:0.2f} mm",
            (center[1], center[0] + 100),
            ha="center",
            va="bottom",
            color="blue",
        )
        # format and legend
        axs[i].axis("off")
        axs[i].legend(loc="upper left")

    # compute the mm to px relation and indicate it in the title
    mm_px = d_mm / (2 * r_px)
    fig.suptitle(f"{mm_px:0.4f} [mm/px]", fontsize=15)

    plt.close(fig)  # Evitar que la figura se dibuje automáticamente

    # global_fig = fig
    # plt.show()

    # Función para graficar las imágenes y los círculos
    def update_figure(cx, cy, r, d):

        # diameter line coordinates
        ycoords = [cy, cy]  # Coordenadas Y de la línea
        xcoords = [cx - r, cx + r]  # Coordenadas X de la línea

        for i in range(len(ims)):
            # update circles
            circles[i].center = (cx, cy)
            circles[i].radius = r
            circles[i].set_label(f"r = {r} px")
            # update the annotation
            axs[i].legend(loc="upper left")

            # update diammeter lines
            lines[i].set_data(xcoords, ycoords)

            # update annotations
            antpx[i].set_text(f"{2*r:0.1f} px")
            antpx[i].set_position((cx, cy - 30))
            antmm[i].set_text(f"{d:0.2f} mm")
            antmm[i].set_position((cx, cy + 100))

            axs[i].legend(loc="upper left")
            # axs[i].legend([circles[i]],[f'r = {r_px} px'], loc='upper left')

        # compute the mm to px relation and indicate it in the title
        mm_px = d / (2 * r)
        fig.suptitle(f"{mm_px:0.4f} [mm/px]", fontsize=15)

        # Redibujar la figura sin volver a renderizar todo
        # fig.canvas.draw_idle()
        with output:
            # store the relation value
            relation["mm_px"] = mm_px
            circle_values["center"] = [cy, cx]
            circle_values["r"] = r

            # display the figure
            clear_output(wait=True)
            display(fig)

    # Función que ejecuta la operación con los valores de los sliders
    def mm_to_px_operation(d, r):
        result = d / (2 * r)  # perform the relation
        print(f"Result: {result} mm/px")
        return result

    # Creación de los sliders
    cx_slider = widgets.IntSlider(
        min=0,
        max=im_shape[1],
        step=1,
        value=center[1],
        description="Center X",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    cy_slider = widgets.IntSlider(
        min=0,
        max=im_shape[0],
        step=1,
        value=center[0],
        description="Center Y",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    r_slider = widgets.IntSlider(
        min=0,
        max=max(im_shape) // 2,
        step=1,
        value=r_px,
        description="Radius",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )

    # Cuadro de texto para el tamaño del diammeter in millimeters
    # d_input = widgets.Text(
    #    value=f'{d_mm}',
    #    placeholder='millimeters',
    #    description='Diammeter [mm]:',
    #    layout=widgets.Layout(width='400px')
    # )
    d_input = widgets.FloatText(
        value=d_mm,  # El valor inicial que se muestra
        description="Diammeter [mm]:",  # Texto de descripción que acompaña al cuadro
        disabled=False,  # Si es True, el widget estará deshabilitado
        placeholder="millimeters",
        continuous_update=False,
        layout=widgets.Layout(width="400px"),
        style={"description_width": "200px"},  # Ampliar solo el área de la descripción
    )

    # Cuadro de texto para el nombre del archivo
    filename_input = widgets.Text(
        value="pixel_to_mm_mean.pdf",
        placeholder="filename",
        description="Filename:",
        continuous_update=False,
        layout=widgets.Layout(width="400px"),
    )

    # Botón para guardar la imagen
    save_button = widgets.Button(
        description="Save Image", layout=widgets.Layout(width="200px")
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
            flua.save_figure(fig, ofname, subdirs=subdir, **fig_kwargs)
            print(f"Imagen guardada como: {ofname}")
        else:
            print("No hay ninguna figura para guardar.")

    save_button.on_click(on_save_button_click)

    # Botón para ejecutar la operación y mostrar resultados
    operation_button = widgets.Button(
        description="Compute mm to px relation", layout=widgets.Layout(width="300px")
    )
    result_value = widgets.FloatText(
        description="Result [mm/px]",
        disabled=True,
        style={"description_width": "100px"},
    )

    # Conectar el botón con la función que ejecuta la operación
    def on_operation_button_click(_):
        result = mm_to_px_operation(d_input.value, r_slider.value)
        result_value.value = result  # Mostrar el resultado en el widget de texto

    operation_button.on_click(on_operation_button_click)

    # Cuadro de texto con instrucciones
    instructions = widgets.HTML(
        value="""
        <h3>Instrucciones:</h3>
        <ul>
            <li>Ajusta el centro y el radio del círculo usando los sliders o ingresando directamente el valor.</li>
            <li>El círculo se dibuja en ambas imágenes simultáneamente para comprobar su coherencia.</li>
            <li>Ingresa el valor asociado (en milimetros) asociado a ese circulo.</li>
            <li>Introduce el nombre del archivo y haz clic en "Guardar Imagen" para guardar la figura.</li>
        </ul>
        """,
        layout=widgets.Layout(width="800px"),
    )

    # Conectar los sliders con la función interactiva
    """
    interactive_plot = widgets.interactive_output(update_figure, {
        'cx': cx_slider, 
        'cy': cy_slider, 
        'r': r_slider,
        'd': d_input
    })
    """

    # Vincular la actualización de la figura a los sliders
    def on_slider_change(change):
        update_figure(cx_slider.value, cy_slider.value, r_slider.value, d_input.value)

    # Conectar los eventos de los sliders a la función de actualización
    cx_slider.observe(on_slider_change, names="value")
    cy_slider.observe(on_slider_change, names="value")
    r_slider.observe(on_slider_change, names="value")
    d_input.observe(on_slider_change, names="value")

    # Controles de widgets
    controls = widgets.VBox(
        [
            instructions,
            d_input,
            cx_slider,
            cy_slider,
            r_slider,  # widgets.HBox([cx_slider, cy_slider, r_slider]),
            widgets.HBox([filename_input, save_button]),
            operation_button,
            result_value,
        ]
    )

    output = widgets.Output()

    # Mostrar los widget y la figura inicial
    display(controls, output)
    update_figure(center[1], center[0], r_px, d_mm)

    # Devolver la figura (por si se quiere usar en otra cosa) y el valor de la relacion en un dictionario
    return circle_values, relation


def inspect_signal_simple(signal_vector, s_param, x_label="", y_label="Signal value"):
    """
    # Currently non used function, prefer the interactive version

    To display the signal vector, an smooting and its derivative.
    The smoothing (by means of subic spline) is performed to better undesrstand
    the tendence of the signal and get a less noisy derivative.

    Parameters
    ----------
    sinal_vector: unidimentional array or list
        signal to be inspected

    s_param: int (positive)
        Smoothing parameter.
        The smoothing increase with this value
        s_param = 0 perfectly fits the data without smoothing.

    x_label: string
        string label of x-axis

    y_label: string
        string label of y-axis

    """

    # create a x vector of the same length as signal
    x = np.arange(len(signal_vector))

    # Ajustar un spline cúbico a los datos
    spline = UnivariateSpline(x, signal_vector, s=50)

    # Calcular la derivada del spline
    dy_dx_spline = spline.derivative()(x)

    fig, axs = plt.subplots(figsize=(10, 6))
    # axs.set_title('mean signal')
    # Gráfico del ajuste
    axs.plot(signal_vector, "k.")
    axs.plot(spline(x), "r-")
    axs.set_xlabel(x_label, color="k")  #'tab:red')
    axs.set_ylabel(y_label, color="k")  #'tab:red')

    # Gráfico de la derivada
    axs2 = axs.twinx()
    axs2.plot(dy_dx_spline, "b-")
    axs2.set_ylabel("derivative", color="tab:blue")
    axs2.tick_params(axis="y", labelcolor="tab:blue")

    # Linea horizontal en el cero
    axs2.axhline(0, color="y")

    # Marcar los puntos críticos
    sign_changes = np.where(np.diff(np.sign(dy_dx_spline)))[0]
    axs2.scatter(
        sign_changes,
        np.zeros(sign_changes.shape),
        marker="x",
        color="y",
        zorder=5,
        label="Critical points",
    )

    # Agregar anotaciones en los puntos críticos
    for xc in sign_changes:
        yc = spline(xc)  # Valor de y en el punto crítico
        axs2.text(
            xc, -0.04, f"{xc:.0f}", fontsize=10, ha="center", va="bottom", color="y"
        )


def inspect_signal(signal_vector, max_s_param=100, x_label="", y_label="Signal value"):
    """
    To display the signal vector, a smoothing and its derivative.
    The smoothing (by means of cubic spline) is performed to better understand
    the tendency of the signal and get a less noisy derivative.

    Parameters
    ----------
    signal_vector: unidimensional array or list
        Signal to be inspected

    x_label: string
        string label of x-axis

    y_label: string
        string label of y-axis


    Returns
    -------
    results: dict
        it contains the spline function and its derivative function
        {'spline': function, 'derivative': function}

    """
    # init the result dictionary
    result = {"spline": None, "derivative": None}

    # create a x vector of the same length as signal
    x = np.arange(len(signal_vector))

    # Crear la figura una sola vez (plot the signal just to init the lines)
    fig, axs = plt.subplots(figsize=(9, 5))

    axs.plot(x, signal_vector, "k.", label="Original Signal")  # the data points
    axs.set_xlabel(x_label, color="k")
    axs.set_ylabel(y_label, color="k")

    (fit,) = axs.plot(x, signal_vector, "r-", label="Smoothed Signal")  # the fit line

    axs2 = axs.twinx()  # twin axis for the derivative
    (dsdx,) = axs2.plot(
        x, signal_vector, "b-", label="Derivative"
    )  # the derivative line
    axs2.set_ylabel("derivative", color="tab:blue")
    axs2.tick_params(axis="y", labelcolor="tab:blue")

    # horizontal line at zero
    axs2.axhline(0, color="y", linestyle="--")

    # critical points
    (cps,) = axs2.plot([], [], "yx", zorder=5, label="Critical points")

    fig.legend(loc="upper left")

    plt.close(fig)  # Evitar que quede repetida

    # **List to store annotations dynamically**
    annotations = []

    def update_annotations(x_positions, min_ax, max_ax):
        """Remove existing annotations and add new ones."""
        # Remove all previous annotations
        for annotation in annotations:
            annotation.remove()
        annotations.clear()

        # define the y axis position of text
        drange = max_ax - min_ax
        ypos = -0.06 * drange

        # Add new annotations
        for xc in x_positions:
            annotation = axs2.text(
                xc, ypos, f"{xc:.0f}", fontsize=10, ha="center", va="bottom", color="y"
            )
            annotations.append(annotation)

    # Función para actualizar la figura con el parámetro de suavizado
    def update_splot(s_param):

        # Fit a cubic spline using s_param
        spline = UnivariateSpline(x, signal_vector, s=s_param)

        # get it derivative evaluated in x vector
        dy_dx_spline = spline.derivative()(x)

        # update them in the figure
        fit.set_data(x, spline(x))
        dsdx.set_data(x, dy_dx_spline)

        # change the y-axis limits
        mind = min(dy_dx_spline)
        maxd = max(dy_dx_spline)

        axs2.set_ylim([mind - 0.1 * maxd, 1.1 * maxd])  # give 0.1 looseness

        # Identificar puntos críticos
        sign_changes = np.where(np.diff(np.sign(dy_dx_spline)))[0]
        # update them
        cps.set_data(sign_changes, np.zeros(sign_changes.shape))

        # **Update critical points annotations dynamically**
        update_annotations(sign_changes, mind, maxd)

        # Redibujar la figura sin recargarla
        with output:
            # update the result dictionary
            result["spline"] = spline
            result["derivative"] = dy_dx_spline

            clear_output(wait=True)  # para eliminar el anterior
            display(fig)

    # Crear el slider para ajustar s_param
    s_param_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max_s_param,
        step=1,
        description="Smooth Parameter",
        continuous_update=False,
        style={"description_width": "120px"},
        layout=widgets.Layout(width="640px"),
    )

    # Cuadro de texto con instrucciones
    instructions = widgets.HTML(
        value="""
        <h3>Instrucciones:</h3>
        <ul>
            <li>Ajusta el parametro de smoothing usando el slider o ingresando directamente el valor.</li>
        </ul>
        """,
        layout=widgets.Layout(width="800px"),
    )

    # Función que se ejecuta al mover el slider (vinculando la actualización de la figura al slider)
    def on_slider_change(change):
        update_splot(s_param_slider.value)

    # Conectar un evento en el slider con la función de actualización
    s_param_slider.observe(on_slider_change, names="value")

    # Controles de widgets
    controls = widgets.VBox([instructions, s_param_slider])

    output = widgets.Output()

    # Mostrar los widget y la figura inicial
    display(controls, output)
    update_splot(0)  # si no se incluye, de todas formas aparecerá al mover los sliders.

    return result


def im_zoom(x_lims, y_lims, image, show_zoom=True, cbar=False, get_cim=False):
    """
    Make a zoom of a region of an image's region of interest

    Parameters
    ----------
        xlims: list
            x-axis limits of the zoomed section.
            e.g. [x_min, x_max]

        ylims: list
            y-axis limits of the zoomed section.
            e.g. [y_min, y_max]

        image: numpy array
            the image array to be display

        show_zoom: boolean
            True to diplay the performed zoom

        cbar: boolean
            True to display the colorbar

        get_cim: boolean
            True to return the cropped image array

    Returns
    -------
        cim: array like
            cropped image array

    """
    x0 = x_lims[0]
    x1 = x_lims[1]
    y0 = y_lims[0]
    y1 = y_lims[1]

    # convert on steps because the rectangle patch definition
    X2R = x1 - x0
    Y2R = y1 - y0

    # crop the image
    cim = image[y0 : y1 + 1, x0 : x1 + 1]  # +1 because the slicing

    if show_zoom:
        ## Perform the plots ##
        fig = plt.figure(figsize=(15, 5))

        # full image
        plt.subplot(121)
        plt.imshow(image)

        # rectangle

        ax = fig.gca()
        # ([x0,y0], width, heigh,...)
        rect = Rectangle(
            (x0, y0), X2R, Y2R, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

        # cropped image
        plt.subplot(122)
        plt.imshow(cim)

        if cbar:
            plt.colorbar()

    # return
    if get_cim:
        return cim


def im_zoom_interact(image, cbar=False, get_cim=False, subdir=""):
    """
    Interactive function to make a zoom of an image's region of interest

    Parameters
    ----------

        image: numpy array
            the image array to be display

        cbar: boolean
            True to display the colorbar

        get_cim: boolean
            True to return the cropped image array

        subdir: string
            Subdirectoy to save the generated figure

    Returns
    -------
        cim: dictionary with
            zoom:
                cropped image array

            x0,y0: list
                top left coordinates of the zoom rectangle

            h,w: list
                heigt and wide of the zoom image

    """

    h_max, w_max = image.shape[0:2]

    x0, y0 = 0, 0
    h0, w0 = int(h_max / 10), int(w_max / 10)

    # crop the image
    cim = image[y0 : y0 + h0, x0 : x0 + w0]

    # init the dictionary with the returned values
    result = {"zoom": cim, "y0,x0": [y0, x0], "h,w": [h0, w0]}

    # Crear figura y ejes para las dos imágenes
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # display the image
    axs[0].imshow(image)  # , cmap='gray')
    axs[0].set_title("Image")

    # dibujar el rectagulo
    # ([x0,y0], width, heigh,...)
    rect = Rectangle(
        (x0, y0), w0, h0, linewidth=1, edgecolor="r", facecolor="none", label="zoom"
    )
    axs[0].add_patch(rect)

    # cropped image
    axs[1].imshow(cim)
    axs[1].set_title("Zoom")

    if cbar:
        axs[1].colorbar()

    # format and legend
    # axs.legend(loc = 'upper left')

    # compute the mm to px relation and indicate it in the title
    # fig.suptitle(f'Zoom', fontsize=15)

    plt.close(fig)  # Evitar que la figura se dibuje automáticamente

    # Función para graficar las imágenes
    def update_figure(y, x, h, w):

        # update the rectangle
        rect.xy = (x, y)
        rect.set_height(h)
        rect.set_width(w)

        # update the annotation
        # rect.set_label(f'r = {r} px')
        # axs[0].legend(loc = 'upper left')

        # update the zoom image
        cim = image[y : y + h, x : x + w]
        axs[1].imshow(cim)

        # fig.suptitle(f'{mm_px:0.4f} [mm/px]', fontsize=15)

        # Redibujar la figura sin volver a renderizar todo
        # fig.canvas.draw_idle()
        with output:
            # store the values
            result["zoom"] = cim
            result["y0,x0"] = [y, x]
            result["h,w"] = [h, w]

            # display the figure
            clear_output(wait=True)
            display(fig)

    # Creación de los sliders
    y0_slider = widgets.IntSlider(
        min=0,
        max=h_max - 2,
        step=1,
        value=y0,
        description="y0 (top)",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    x0_slider = widgets.IntSlider(
        min=0,
        max=w_max - 2,
        step=1,
        value=x0,
        description="x0 (left)",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    h_slider = widgets.IntSlider(
        min=1,
        max=h_max - y0_slider.value,
        step=1,
        value=h0,
        description="Height",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    w_slider = widgets.IntSlider(
        min=1,
        max=w_max - x0_slider.value,
        step=1,
        value=w0,
        description="Wide",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )

    # Cuadro de texto para el nombre del archivo
    filename_input = widgets.Text(
        value="Zoom_01.pdf",
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
            flua.save_figure(fig, ofname, subdirs=subdir, **fig_kwargs)
            print(f"Imagen guardada como: {ofname}")
        else:
            print("No hay ninguna figura para guardar.")

    save_button.on_click(on_save_button_click)

    # Cuadro de texto con instrucciones
    instructions = widgets.HTML(
        value="""
        <h3>Instructions:</h3>
        <ul>
            <li>Choose top-left (y0,x0), heigh and wide values of the rectangle zoom area. </li>
            <li>Input desired filename and click "Save Image" to store the figure.</li>
        </ul>
        """,
        layout=widgets.Layout(width="800px"),
    )

    # Función para actualizar el valor máximo de los slider h y w
    def update_h_max(change):

        h_slider.max = h_max - change["new"]  # update the max value

    def update_w_max(change):

        w_slider.max = w_max - change["new"]

    # observe the value and update in the other slider
    y0_slider.observe(update_h_max, names="value")
    x0_slider.observe(update_w_max, names="value")

    # Vincular la actualización de la figura a los sliders
    def on_slider_change(change):
        update_figure(y0_slider.value, x0_slider.value, h_slider.value, w_slider.value)

    # Conectar los eventos de los sliders a la función de actualización
    y0_slider.observe(on_slider_change, names="value")
    x0_slider.observe(on_slider_change, names="value")
    h_slider.observe(on_slider_change, names="value")
    w_slider.observe(on_slider_change, names="value")

    # Controles de widgets
    controls = widgets.VBox(
        [
            instructions,
            widgets.HBox([y0_slider, x0_slider]),
            widgets.HBox([h_slider, w_slider]),
            widgets.HBox([filename_input, save_button]),
        ]
    )

    output = widgets.Output()

    # Mostrar los wodget y la figura inicial
    display(controls, output)
    update_figure(y0_slider.value, x0_slider.value, h_slider.value, w_slider.value)

    # return
    if get_cim:
        return result


def im_circle_interact(image, center0=None, cbar=False, subdir="", lw=1):
    """
    Interactive function to define a circle to limit the colonies search area.
    # Add the option to define a rectangle

    Parameters
    ----------

        image: numpy array
            the image array to be display

        center0: list or array
            initial center position [y,x]

        cbar: boolean
            True to display the colorbar

        subdir: string
            Subdirectoy to save the generated figure

        lw: int
            plots line width. It also determines the center/centroid dot size.

    Returns
    -------
        cim: dictionary with

            x0,y0: list
                center coordinates of the circle

            r: numeric
                radious of the circle

    """

    h, w = image.shape[0:2]
    if center0 is None:
        y0, x0 = int(h / 2), int(w / 2)
    else:
        y0, x0 = flua.round_up(center0[0]), flua.round_up(center0[1])

    r0 = min([y0, x0])
    rmax = int(max([h, w]) / 2)  # to limit the slider

    text_off_y = int(0.05 * h)  # center text position offset in y-axis

    # init the dictionary with the returned values
    result = {"y0,x0": [y0, x0], "r": r0}

    # Crear figura y ejes para las dos imágenes
    fig, ax = plt.subplots(figsize=(12, 6))

    # display the image
    ax.imshow(image)  # , cmap='gray')
    ax.set_title("Image")

    # Draw the initial circle
    circle = Circle((x0, y0), r0, color="r", fill=False, lw=1, label=f"r = {r0} px")
    ax.add_artist(circle)

    # Draw the center
    (center_dot,) = ax.plot((x0, y0), "or", markersize=lw + 1, linewidth=lw)
    center_text = ax.annotate(
        f"[{y0},{x0}]", (x0, y0 - text_off_y), ha="center", va="bottom", color="red"
    )

    # format and legend
    ax.legend(loc="upper left")

    plt.close(fig)  # Evitar que la figura se dibuje automáticamente

    # Función para graficar las imágenes
    def update_figure(y, x, r):

        # update circles
        circle.center = (x, y)
        circle.radius = r
        circle.set_label(f"r = {r} px")

        # Update center dot and text
        center_dot.set_data([x], [y])
        center_text.set_text(f"[{y},{x}]")
        center_text.set_position((x, y - text_off_y))

        # update the annotation
        # rect.set_label(f'r = {r} px')
        ax.legend(loc="upper left")

        # Redibujar la figura sin volver a renderizar todo
        # fig.canvas.draw_idle()
        with output:
            # store the values
            result["y0,x0"] = [y, x]
            result["r"] = r

            # display the figure
            clear_output(wait=True)
            display(fig)

    # Creación de los sliders
    y0_slider = widgets.IntSlider(
        min=0,
        max=h - 1,
        step=1,
        value=y0,
        description="center Y",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    x0_slider = widgets.IntSlider(
        min=0,
        max=w - 1,
        step=1,
        value=x0,
        description="center X",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    r_slider = widgets.IntSlider(
        min=1,
        max=rmax,
        step=1,
        value=r0,
        description="Radius",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )

    # Cuadro de texto para el nombre del archivo
    filename_input = widgets.Text(
        value="Search_area.pdf",
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
            flua.save_figure(fig, ofname, subdirs=subdir, **fig_kwargs)
            print(f"Imagen guardada como: {ofname}")
        else:
            print("No hay ninguna figura para guardar.")

    save_button.on_click(on_save_button_click)

    # Cuadro de texto con instrucciones
    instructions = widgets.HTML(
        value="""
        <h3>Instructions:</h3>
        <ul>
            <li>Choose the values for the circle center (Y,X) and radius (in pixels). </li>
            <li>Input desired filename and click "Save Image" to store the figure.</li>
        </ul>
        """,
        layout=widgets.Layout(width="800px"),
    )

    # Vincular la actualización de la figura a los sliders
    def on_slider_change(change):
        update_figure(y0_slider.value, x0_slider.value, r_slider.value)

    # Conectar los eventos de los sliders a la función de actualización
    y0_slider.observe(on_slider_change, names="value")
    x0_slider.observe(on_slider_change, names="value")
    r_slider.observe(on_slider_change, names="value")

    # Controles de widgets
    controls = widgets.VBox(
        [
            instructions,
            y0_slider,
            x0_slider,
            r_slider,
            widgets.HBox([filename_input, save_button]),
        ]
    )

    output = widgets.Output()

    # Mostrar los wodget y la figura inicial
    display(controls, output)
    update_figure(y0_slider.value, x0_slider.value, r_slider.value)

    # return the circle values
    return result


def display_im_mask(
    image,
    mask,
    over_class="all",
    alpha=0.5,
    cmap_name="viridis",
    legend=True,
    ofname=None,
    **kwargs,
):
    """
    display the mask superposed over the input image using overlay transparency
    accord the alpha parameter.

    Parameters
    ----------
    image : np.ndarray
        Image as numpy array, monochromatic (2D) or RGB (3D).
        its data type has to be in uint8 format.

    mask : np.ndarray (numeric or boolean)
        2D Binary (2 cathegories) or multicathegory mask array with the
        same size as the image.
        (in case of multycathegory, each one is indicated by a integer value)

    over : 'all', bool, list
        Modo de superposición:
        - 'all': Aplica color a toda la imagen.
        - True/False/0/1: Apply the color just to the mask indicated value
                            keeping the color of the rest. e.g. over = True will
                            color the mask pixels equal to True.

    alpha : float
        Overlay transparency (0 = full transparent, 1 = full opaque).

    cmap_name : str, optional
        Name of the matplotlib colormap to be used to represent the mask cathegories

    legend : bool, optional
        if True, display the legend with the colors assigned to each mask class.

    ofname: string
        if given, the figure is stored under that name.
        More options with **kwargs  (see flua.save_fig())
    """

    # get the colormap and normalize the mask
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=mask.min(), vmax=mask.max())

    # covert the mask to a color image accord the chosen colormap
    mask_colored = cmap(norm(mask))  # RGBA mask

    # perform image convertion, store its information to recover it at the end
    im_dtype = image.dtype

    if im_dtype == np.uint8:
        image = (
            image.astype(np.float32) / 255.0
        )  # convert uint8 [0,255] to float32 [0,1]
        scale_back = 255  # scale factor to recover the image after processing

    elif np.issubdtype(im_dtype, np.floating) and image.max() <= 1.0:
        image = image.astype(np.float32)
        scale_back = 1

    else:
        raise ValueError(
            "Incompatible image format. Valid options are uint8 [0,255] or float [0,1]."
        )

    # if image is monochromatic (2D) expand it to 3D by copying it channel to the other 2
    # it is necessary for to create the supoerposed image (3D)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)  # Convert (N, M) → (N, M, 3)

    # convert image to float to perform the overlay computations
    image = image.astype(np.float32)

    # create the overlayed image
    if over_class == "all":

        overlay = (1 - alpha) * image + alpha * mask_colored[..., :3]
        # overlay = (1 - alpha) * image + alpha * (mask_colored[..., :3] * 255)  # * 255 to convert the colormap to uint8 range
        legend_classes = np.unique(mask)  # include all classes in plot legend

    elif isinstance(over_class, (int, bool, list, np.integer, np.ndarray)):

        # create a copy of the image
        overlay = np.copy(image)

        # convert to 1D array for code generalization
        over_class = np.atleast_1d(
            over_class
        )  # Convierte cualquier entrada a un array de 1D

        # create a combined boolean mask
        mask_combined = np.isin(mask, over_class)

        # overlay just the indicated classes over the image
        overlay[mask_combined] = (1 - alpha) * image[
            mask_combined
        ] + alpha * mask_colored[mask_combined, :3]
        # overlay[mask_combined] = (1 - alpha) * image[mask_combined] + alpha * (mask_colored[mask_combined, :3] * 255) # * 255 to convert the colormap to uint8 range

        # include just the selected classes in plot legend
        legend_classes = np.unique(mask[mask_combined])

    else:
        raise ValueError(
            "'over_class' parameter have to be 'all' or one mask class value (e.g. 0)"
        )

    # convert again to original format and ensure is inside the data type range by clipping
    # overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay = np.clip(overlay * scale_back, 0, scale_back).astype(im_dtype)

    # Display the result superposed image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(overlay)
    ax.set_title("Superposed mask")
    ax.axis("off")

    # Add the legend if indicated
    if legend and len(legend_classes) > 0:

        patches = [
            mpl.patches.Patch(color=cmap(norm(label)), label=f"Class {label}")
            for label in legend_classes
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=10, title="Clases")

    # save the figure image file if indicated
    if ofname is not None:

        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        fig_kwargs.update(kwargs)  # join with the input functions kwargs
        flua.save_figure(fig, ofname, **fig_kwargs)

    plt.show()


def plot_blobs(
    blobs,
    image,
    cmap="gray",
    title=None,
    axis="off",
    ccolor="r",
    clw=0.5,
    show_ids=True,
    text_displace=[-1, -1],
    ids_color="white",
    roi_mode="square",
    zoom_lims=None,
    ofname=None,
    fformat=".pdf",
    overwrite=False,
):
    """

    Parameters
    ----------
    blobs: array (Nx3)
        contains the (y,x) center position and radius of each blob
        for each of N colonies.
        This values should be relative to the input image.

        Structure for the 'i' element: [yi, xi, ri]

    image: str or array
        image filename path or image array.
        for path, the image is loaded.
        The detected blobs are displayed over it.
        This image could be mono or multichannel.

    cmap: colormap name or cmap object
        colormap used to display the image in case it is monochromatic.

    axis: str
        'on' to display the plot axis
        'off' to not display them

    ccolor: any matplotlib color indicator
         color used to display the circles over the image

    clw: numeric
        circles linewith to display them over the image

    show_ids: bool
        if True, display the IDs (i.e. position in the blobs array) of
        each blob together its plotted circle.

    text_displace: list with two numbers
        It is only funcional if show_ids is True.
        The factors applied to displace the ID text from the circles center.
        This factor is multiplied by the radius of each colony.
        e.g. text_displace = [-2,1] --> text will be displaced [-2*ri, 1*ri],
        where ri is the radius of the circle of colony i.

    ids_color: any matplotlib color indicator
        color used to display the blobds ids over the image

    roi_mode : str
        the mode to display the roi over the image. options: 'square' or 'circle'
        default is 'square'

    zoom_lims: list
        list with the zoom coordinates to crop the image.
        Structure: [[ymin,ymax],[xmin,xmax]]

    ofname: str
        if given, save the image with the circles over it (image+blobs+ID) with this name.

    fformat: str
        file format extension of the output stored image.

    overwrite: bool
        If False, you will be asked prior to overwrite the file
        If True, the files will be overwrittem directly.

    """

    if type(image) == str:

        imname = image  # store the image name to use in title

        # Read the source file image
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

        # in case of multicromatic image, convert to RGB
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # perform zoom if indicated
    ylims = [0, image.shape[0]]
    xlims = [0, image.shape[1]]

    if zoom_lims is not None:
        ylims = zoom_lims[0]
        xlims = zoom_lims[1]
        image = im_zoom(ylims, xlims, image, show_zoom=False)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    if image.ndim == 2:
        img = ax.imshow(image, cmap=cmap)
        fig.colorbar(img, fraction=0.04)

    else:
        img = ax.imshow(image)

    if title is None:
        title = "Over " + imname

    ax.set_title(f"{title}")
    ax.axis(axis)  # to display or not the axis

    for i in range(blobs.shape[0]):

        yi = blobs[i, 0] - ylims[0]  # subtract the zoom offset if it's the case
        xi = blobs[i, 1] - xlims[0]
        ri = blobs[i, 2]

        if roi_mode == "circle":
            # circles syntax is: (x,y),r
            circle = plt.Circle((xi, yi), ri, color=ccolor, fill=False, lw=clw)
            ax.add_artist(circle)
        else:
            # here we use the discretized rectangle instead of the exact blob float values
            x0 = int(np.floor(xi - ri + 0.5))
            y0 = int(np.floor(yi - ri + 0.5))
            wh = int(np.floor(2 * ri + 0.5))
            rectang = Rectangle(
                (x0, y0), wh, wh, linewidth=1, edgecolor="r", facecolor="none"
            )
            ax.add_artist(rectang)

        if show_ids:

            # coordinates to put the text that indicates the point xi,yi
            text_offset_i = (xi + ri * text_displace[0], yi + ri * text_displace[1])

            # attach the ID label to each colony
            ax.annotate(
                str(i),
                xy=(xi, yi),
                xytext=text_offset_i,
                xycoords="data",
                textcoords="data",
                ha="right",
                va="bottom",
                color=ids_color,
            )
            # 'data' to use the same data coordinates values to situate the text
            # ha='right' and va='bottom', to situate the "right bottom" of the text
            # in the xi,yi point, which then is displaced accord text_offset_i

    # save the figure image file if indicated
    if ofname is not None:
        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        flua.save_figure(fig, ofname, fformat, overwrite=overwrite, **fig_kwargs)

    plt.show()


def plot_ROIs(
    rois,
    image,
    cmap="gray",
    title=None,
    axis="off",
    rcolor="r",
    lw=1,
    show_ids=True,
    text_displace=[-1, -1],
    ids_color="white",
    zoom_lims=None,
    ofname=None,
    fformat=".pdf",
    overwrite=False,
):
    """

    Parameters
    ----------
    rois: dict
        dict of ROIs, with the IDs as keys

    image: str or array
        image filename path or image array.
        for path, the image is loaded.
        The detected blobs are displayed over it.
        This image could be mono or multichannel.

    cmap: colormap name or cmap object
        colormap used to display the image in case it is monochromatic.

    axis: str
        'on' to display the plot axis
        'off' to not display them

    rcolor: any matplotlib color indicator
         color used to display the rectangles over the image

    lw: numeric
        rectangle linewith to display them over the image

    show_ids: bool
        if True, display the IDs (i.e. position in the blobs array) of
        each blob together its plotted circle.

    text_displace: list with two numbers
        It is only funcional if show_ids is True.
        The x and y pixels displacement of the text in the image.
        e.g. text_displace = [-2,1] --> text will be displaced -2 pixels
        in the x-axis and 1 pixels in the y-axis.
        considerations:
            negative values in y-axis means to move up
            negative values in x-axis means to move left

    ids_color: any matplotlib color indicator
        color used to display the blobds ids over the image

    zoom_lims: list
        list with the zoom coordinates to crop the image.
        Structure: [[ymin,ymax],[xmin,xmax]]

    ofname: str
        if given, save the image with the circles over it (image+blobs+ID) with this name.

    fformat: str
        file format extension of the output stored image.

    overwrite: bool
        If False, you will be asked prior to overwrite the file
        If True, the files will be overwrittem directly.

    """

    if type(image) == str:

        imname = image  # store the image name to use in the title

        # Read the source file image
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

        # in case of multicromatic image, convert to RGB
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # perform zoom if indicated
    ylims = [0, image.shape[0]]
    xlims = [0, image.shape[1]]

    if zoom_lims is not None:
        ylims = zoom_lims[0]
        xlims = zoom_lims[1]
        image = im_zoom(ylims, xlims, image, show_zoom=False)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    if image.ndim == 2:
        img = ax.imshow(image, cmap=cmap)
        fig.colorbar(img, fraction=0.04)

    else:
        img = ax.imshow(image)

    if title is None:
        title = "Over " + imname

    ax.set_title(f"{title}")
    ax.axis(axis)  # to display or not the axis

    for roi in rois.values():
        # get the roi limits
        roi_yl = roi.ylims
        roi_xl = roi.xlims

        # get the roi height and wide
        roi_h = roi.hw[0]
        roi_w = roi.hw[1]

        # get the roi left bottom corner
        x0 = roi_xl[0] - xlims[0]  # subtract the zoom offset if it's the case
        y1 = roi_yl[1] - ylims[0]

        rectang = plt.Rectangle(
            (x0, y1), roi_w, -roi_h, linewidth=lw, edgecolor=rcolor, facecolor="none"
        )
        ax.add_artist(rectang)

        if show_ids:

            # coordinates to put the text that indicates the point xi,yi
            text_offset_i = (
                x0 + roi_w + text_displace[0],
                y1 - roi_h + text_displace[1],
            )

            # attach the ID label to each colony
            ax.annotate(
                str(roi.id),
                xy=(x0, y1),
                xytext=text_offset_i,
                xycoords="data",
                textcoords="data",
                ha="left",
                va="bottom",
                color=ids_color,
            )
            # 'data' to use the same data coordinates values to situate the text
            # ha='right' and va='bottom', to situate the "right bottom" of the text
            # in the xi,yi point, which then is displaced accord text_offset_i

    # save the figure image file if indicated
    if ofname is not None:
        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        flua.save_figure(fig, ofname, fformat, overwrite=overwrite, **fig_kwargs)

    plt.show()


def plot_roi_frame(
    roi,
    col_id=None,
    channels=None,
    n_frame=-1,
    yc_line=False,
    xc_line=False,
    ofname=None,
    fformat=".pdf",
    **kwargs,
):
    """
    to display a roi frame for given channels

    The center position if obtained from roi attribute, it is not computed here.

    Parameters:
    -----------
    roi: roi object or dict
        if a dict, you also need to give a col_id to select that one

    channels: list
        list of roi data channel names to use

    col_id: integer
        colony id used to select the roi in case input roi is a list.

    nframe: integer
        frame to display

    yc_line: bool
        if True, display a horizontal line in the colony center

    xc_line: bool
        if True, display a vertical line in the colony center

    """
    # get the arguments of plot_roi_frame to filter them from **kwargs
    prf_params = inspect.signature(plot_roi_frame).parameters

    # filter kwargs for axhline and axvline
    axl_args = {k: v for k, v in kwargs.items() if k not in prf_params}

    # get the roi
    roi = flua.get_roi(roi, col_id)

    # verifyt type values:
    if type(channels) == str:
        channels = [channels]

    n_frame = int(np.floor(n_frame + 0.5))

    if channels is None:
        channels = roi.channels

    if n_frame == -1:
        n_frame = roi.nframes - 1

    # get number of channels and init the figure
    nchans = len(channels)

    fig, axs = plt.subplots(1, nchans, figsize=(4 * (nchans), 4), layout="constrained")

    if nchans == 1:
        axs = [axs]

    for i in range(nchans):

        chani = channels[i]
        roi_c = roi.data[chani]

        if roi_c.ndim == 3:

            imi = axs[i].imshow(roi_c[:, :, n_frame])

        elif roi_c.ndim == 2:

            imi = axs[i].imshow(roi_c)
            n_frame = None

        else:
            print(f"invalid roi.data[{chani}] dimentions")
            return

        if yc_line:
            axs[i].axhline(y=int(np.floor(roi.center[0] + 0.5)), **axl_args)
        if xc_line:
            axs[i].axvline(x=int(np.floor(roi.center[1] + 0.5)), **axl_args)

        axs[i].set_title(f"{chani} channel")
        fig.colorbar(imi, fraction=0.04)

    fig.suptitle(f"Colony {roi.id}, frame {n_frame}", fontsize=12)

    # save the image file if indicated
    if ofname is not None:

        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        flua.save_figure(fig, ofname, fformat, overwrite=overwrite, **fig_kwargs)

    plt.show()


def roi_kgraph(
    rois,
    col_id,
    channels,
    rows=None,
    columns=None,
    xltext="frame",
    add_center=True,
    ofname=None,
    fformat=".pdf",
    overwrite=False,
    getvalues=False,
):
    # add **kwargs to something?
    """
    to create a roi kymograph
    it is assumed each roi.data[chan] has dimentions
    three dimentions [n,m, frames]
    By default it display the kemograph of the colony center, at least other
    rows or columns are indicated.

    Parameters
    ----------
    channels: list
    xltext: str
        x label text
    add_center: bool
        if True, a line is plot at the center position of the colony
        in the kemograph

    ofname: str
        base name. More text is added automatically

    getvalues: bool
        if True, the dictionary with the kemograph series is returned

    overwrite: bool
        If False, you will be asked prior to overwrite the file
        If True, the files will be overwrittem directly.

    Returns
    -------
    [optional], make getvalues = True, to obtain them

    kgs: dict
        dictionary with the data series of the kemographs requested.

    """
    # get the roi
    roi = flua.get_roi(rois, col_id)

    # get the number of channels
    nc = len(channels)

    # init the dictionary to store each data serie for the kemographs
    kgs = dict()

    yci = int(np.floor(roi.center[0] + 0.5))
    xci = int(np.floor(roi.center[1] + 0.5))

    if rows is None:
        rows = [yci]

    if columns is None:
        columns = [xci]

    # horizontal line of the ROI
    for row in rows:
        rname = f"row_{row}"
        kgs[rname] = dict()

        for chan in channels:

            try:
                kgs[rname][chan] = roi.data[chan][row, :, :]

            except:
                raise Exception(
                    f"row {row} cannot be obtained from ROI {col_id} data[{chan}]"
                )

        # create the figure
        fig, axs = plt.subplots(1, nc, figsize=(4 * nc, 2), layout="constrained")
        fig.suptitle(f"colony {col_id}, row {row}", fontsize=12)

        if nc == 1:
            axs = [axs]

        for i in range(nc):
            # display the kemogrph image
            imi = axs[i].imshow(kgs[rname][chan])

            if add_center:
                axs[i].axhline(y=yci)

            if i == 0:
                axs[i].set_ylabel("y-axis ROI")

            axs[i].set_xlabel(xltext)
            axs[i].set_title(f"channel {chan}, horizontal transect")
            fig.colorbar(imi, fraction=0.03)

            # save the image file if indicated
            if ofname is not None:
                filename = f"{ofname}kg_{chan}_id{col_id}_row{row}"
                fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
                flua.save_figure(
                    fig, filename, fformat, overwrite=overwrite, **fig_kwargs
                )

            plt.show()

    # vertical line of the ROI
    for column in columns:

        cname = f"column_{column}"
        kgs[cname] = dict()

        for chan in channels:

            try:
                kgs[cname][chan] = roi.data[chan][:, column, :]

            except:
                raise Exception(
                    f"row {row} cannot be obtained from ROI {col_id} data[{chan}]"
                )

        # create the figure
        fig, axs = plt.subplots(1, nc, figsize=(4 * nc, 2), layout="constrained")
        fig.suptitle(f"colony {col_id}, column {column}", fontsize=12)

        if nc == 1:
            axs = [axs]

        for i in range(nc):
            # display the kemogrph image
            imi = axs[i].imshow(kgs[cname][chan])

            if add_center:
                axs[i].axhline(y=xci)

            if i == 0:
                axs[i].set_ylabel("x-axis ROI")

            axs[i].set_xlabel(xltext)
            axs[i].set_title(f"channel {chan}, vertical transect")
            fig.colorbar(imi, fraction=0.03)

            # save the image file if indicated
            if ofname is not None:
                ofname = f"{ofname}kg_{chan}_id{col_id}_col{column}"
                fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
                flua.save_figure(
                    fig, ofname, fformat, overwrite=overwrite, **fig_kwargs
                )

            plt.show()

    # return the values if indicated
    if getvalues:
        return kgs


def display_control_regime(
    dataset, attr="control_regime", time_key="T", t0=0, colors=None, cmap="tab10"
):
    """
    Plot the control regime

    Parameters
    ----------

    dataset: Dataset object
        The dataset which control regime will be display

    attr: str
        The attribute of the dataset where the control regime is defined

    time_key: str
        The key of the time list in the dataset

    t0: float
        initial time of the control regime

    colors: dict
        The colors of each control regime. (the keys are the control regimen names)

    cmap: str
        The color map to be used if no specific colors are provided

    """
    # get the regime
    regime = getattr(dataset, attr)

    # built the time vector
    times = [t0]

    # append two values for each intermediate limit
    for t in regime[time_key][:-1]:
        times.append(t * 0.99)
        times.append(t * 1.01)

    # append the last limit
    times.append(regime[time_key][-1])

    # built the series
    control_signals = flua.generate_control_signal(regime, times, time_key)
    cskeys = list(control_signals.keys())

    # get the colors to plot
    if colors is None:
        colors = get_colors(cmap, dkeys=cskeys)

    # make the figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for cs_name in cskeys:
        cs = control_signals[cs_name]
        ax.plot(times, cs, "--", label=cs_name, color=colors[cs_name])

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Power [%]")
    ax.set_title("Control Regime")
    ax.legend()


def plot_frames_displacement(values, ofname=None, **kwargs):
    """
    to plot frames displacement based on centroids, center and threshold values
    obtained with flua.data_thr_contour()

    Parameters
    ----------
    values: dict
        It should contain at least:

        'frames': np.array
            frame number actually used

        'centroids': np.array
            the centroid coordinates for each contour frame
            centroids[0,:] = [yc0,xc0]
            centroids[:,0] = [yc0,yc1,..,ycn]

        'centers': np.array
            the center coordinates the eclosed circle asociated to the contour
            of each frame
            center[0,:] = [yc0,xc0]
            center[:,0] = [yc0,yc1,..,ycn]

        'r': np.array
            enclosed circle radius values

        'thresholds': np.array
            the Otsu threshold value for each frame

    ofname: string
        if given, the figure is stored under that name. More options
        with **kwargs  (see flua.save_fig())

    """
    # Expected values elements
    # values['frames']
    # values['centroids']
    # values['centers']
    # values['r']
    # values['thresholds']

    # define some colors
    # red, blue, teal, gold
    colors = [160, 0, 0], [26, 128, 187], [41, 140, 140], [241, 162, 38]
    colors = np.asarray(colors) / 255

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), layout="constrained")

    ####  y coordinates values ####
    (l1,) = axs[0].plot(
        values["frames"],
        values["centroids"][:, 0],
        label="centroid",
        color=colors[0],
        marker=".",
        ls="",
    )
    (l2,) = axs[0].plot(
        values["frames"],
        values["centers"][:, 0],
        label="center",
        color=colors[1],
        marker=".",
        ls="",
    )
    axs[0].set_title("y-coordinates")
    axs[0].set_ylabel("y position")
    # axs[0].set_xlabel('frame')
    axs[0].legend()

    ####  y coordinates values ####
    axs[1].plot(
        values["frames"], values["centroids"][:, 1], color=colors[0], marker=".", ls=""
    )
    axs[1].plot(
        values["frames"],
        values["centers"][:, 1],
        color=colors[1],
        marker=".",
        ls="",
    )
    axs[1].set_title("x-coordinates")
    axs[1].set_ylabel("x position")
    # axs[0].set_xlabel('frame')
    axs[1].legend(handles=[l1, l2])

    ####  Threshold  ####
    (l3,) = axs[2].plot(
        values["frames"],
        values["thresholds"],
        label="Threshold",
        color=colors[2],
        marker=".",
        ls="",
    )
    axs[2].set_title("Thresholds and Circle")
    axs[2].set_ylabel("Threshold value [px]", color=colors[2])
    axs[2].set_xlabel("Frame")
    axs[2].tick_params(axis="y", labelcolor=colors[2])

    ####  Enclosed Circle radius ####
    axs_r = axs[2].twinx()
    (l4,) = axs_r.plot(
        values["frames"],
        values["r"],
        label="circle radius",
        color=colors[3],
        marker=".",
        ls="",
    )
    axs_r.set_ylabel("radius", color=colors[3])  #'tab:orange'
    axs_r.tick_params(axis="y", labelcolor=colors[3])

    axs[2].legend(handles=[l3, l4])
    # fig.legend(handles=[l1,l2,l3,l4], loc = 'lower center') #loc = 'upper right' 'lower center'

    plt.show()

    # save the image file if indicated
    if ofname is not None:
        ofname += f""
        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        # fig_kwargs.update(kwargs)  # join with the input functions kwargs
        flua.save_figure(fig, ofname, **fig_kwargs)


def plot_contours_size(contours, thresholds, im_title="", ofname=None, **kwargs):
    """
    to plot the contour size of each given contour associated to the given
    thresholds.
    size in the sense of the pixel count in the contour.

    Parameters
    ----------
    contours: list
        list with the contours.
        It has to be the same length as threshold list

    thresholds: list
        list with the threshold associated to each contour in contours.

    ofname: string
        if given, the figure is stored under that name. More options
        with **kwargs  (see flua.save_fig())

    Return
    ------
    cont_lengths: list
        computed contours lengths
    perimeters: list
        computed contours perimeters
    """

    # Initialize the contour length list and perimeters list
    cont_lengths = []
    perimeters = []

    for contour in contours:
        cont_lengths.append(len(contour))
        perimeter = cv2.arcLength(
            contour, True
        )  # True because contours are closed arcs
        perimeters.append(perimeter)

    # Make the figure
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, cont_lengths, color="r", marker=".")
    ax.set_xlabel("Threshold value [px]")
    ax.set_ylabel("Contour size [px]")
    ax.set_title(im_title)

    # Save the figure file if indicated
    if isinstance(ofname, str):
        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        fig_kwargs.update(kwargs)  # join with the input function's kwargs
        flua.save_figure(fig, ofname, **fig_kwargs)

    return cont_lengths, perimeters


def rois_thr_contours(rois, thr_step, thr_min, ofname=None, **kwargs):
    """
    to plot the contour size of each given contour asociated to the given
    threhsolds.
    size in the sense of the pixel count in the contour.

    Parameters
    ----------
    contours: list
        list with the contours.
        It has to be the same length as threshold list

    thresholds: list
        list with the threshold asociated to each contour in contours.

    ofname: string
        if given, the figure is stored under that name. More options
        with **kwargs  (see flua.save_fig())

    Return
    ------
    cont_lengths: list
        computed contours lenghts

    """

    # init the contour length list
    cont_lengths = []

    for contour in contours:

        cont_lengths.append(len(contour))

    # make the figure
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(thresholds, cont_lengths, color="r", marker=".")

    ax.set_xlabel("Threshold value [px]")
    ax.set_ylabel("Contour size [px]")
    ax.set_title(im_title)

    # save the figure file if indicated
    if type(ofname) is str:

        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        fig_kwargs.update(kwargs)  # join with the input functions kwargs
        flua.save_figure(fig, ofname, **fig_kwargs)

    return cont_lengths


def plot_frame_radial_signal(
    roi,
    d_chan,
    frames=None,
    control_regime=None,
    s_param=900,
    tkey="T",
    colony_key="colony",
    text_x={},
    text_y={},
    text_colors={},
    ofname=None,
    subdirs="",
    file_ext="png",
):
    """

     Plot the mean radial signal for a given colony frame(s)

     # prpath + '/ROIS/ID{roi.id:03}/RMS_serie/RMG_{frame:03d}

     Parameters
     ----------

     roi: ROI object
         ROI object to be plotted

     d_chan: str
         distance channel name to be used

     frames: list
         list of frames to be plotted. It couls be just one.
         if None, it uses all the frames

    control_regime: dict
         control regime schedule. It is used to get the control signal power
         in each frame and plot it.
         if None, no control signals power are plotted

    s_param: int
         smoothing parameter for the spline

     tkey: str
         key of time serie in control signal

     colony_key: str
         key of the colony text parameters to annotate the figure with the colony ID
         It has to be the dictinary key used for text_x, text_y and text_colors.

     text_x: dict
         dict with the x axis positions for each text
         e.g. {'colony': 200, 'R':200, 'G': 200}

     text_y: dict
         dict with the y axis positions for each text
         e.g. {'colony': 180, 'R':170, 'G': 160}

     text_colors: dict
         dict with the colors for each text
         e.g. {'colony': 'k', 'R': 'r', 'G': 'g'}

     ofname: str
         output filename base pattern. The code adds the frame number at the end of this base
         as ofname += f'{frame:03d}'

     subdirs: str
         output subdirectory base. the code adds the roi ID to the folder as
         subdirs += f'{roi.id:03}/'

     file_ext: str
         output file extension

     Returns
     -------
     None
    """
    # conver to list if necessary
    if type(frames) != list:
        frames = [frames]

    max_nframes = roi.data[d_chan].shape[2]
    all_frames = np.arange(max_nframes)

    if frames is None:
        frames = all_frames

    ### get the colony maximum signal and radial distance to set the plot axis ###
    # get the max signal
    max_signal = 0

    for frame in all_frames:  # it has to be all_frames, not the input frames
        try:
            max_frame_signal = roi.rms[frame][1].max()

            # replace the value if greater
            if max_frame_signal > max_signal:
                max_signal = max_frame_signal

        except:
            pass

    # get the maximum radial distance
    max_dist = roi.data[d_chan].max()

    # print(f'Maximum colony radius: {max_dist}')
    # print(f'Maximum colony signal: {max_signal}')

    # get the control signal names
    control_names = []
    if control_regime is not None:
        control_names = [cname for cname in control_regime.keys() if cname != tkey]

    # create default values for text positions if no values were indicated
    if text_x == {}:

        text_x[colony_key] = 0.95 * max_dist

        for cname in control_names:
            text_x[cname] = 0.95 * max_dist

    if text_y == {}:

        text_y[colony_key] = 0.85 * max_signal

        count = 1
        for cname in control_names:
            text_y[cname] = 0.85 * max_signal - count * 10
            count += 1

    if text_colors == {}:
        text_colors[colony_key] = "k"

        for cname in control_names:
            text_colors[cname] = "b"

    # Make a figure for each frame
    for frame in frames:

        # get the frame time
        ftime = roi.times["W"][frame]

        # get the control signals
        if control_regime is not None:
            control_signals = flua.generate_control_signal(
                control_regime, [ftime], tkey
            )

        # get the frame values
        dms_values = roi.rms[frame]  # [distances, averages(d), std(d)]

        # assign to variables for clarity
        distances = dms_values[0]
        signal_avgs = dms_values[1]
        signal_stds = dms_values[2]

        # create the figure
        fig = plt.figure(figsize=(10, 10))

        # try to fit the spline
        try:
            spline = UnivariateSpline(
                distances, signal_avgs, s=s_param
            )  # 's' smoothing parameter
            smooth_dist = spline(distances)
            plt.plot(distances, smooth_dist, "gray", label="smooth")
        except:
            pass
        # plot the average signal values for each radial distance
        plt.plot(distances, signal_avgs, "g.", label="mean")

        # plot the standard deviation
        plt.fill_between(
            distances,
            signal_avgs - signal_stds,
            signal_avgs + signal_stds,
            color="g",
            alpha=0.1,
            label=f"std",
        )

        # anotate the colony id
        try:
            xt, yt = text_x[colony_key], text_y[colony_key]
            plt.annotate(
                f"{colony_key} {roi.id}",
                (0, 0),
                xytext=(xt, yt),
                xycoords="data",
                fontsize=20,
                textcoords="data",
                ha="left",
                va="top",
                color=text_colors[colony_key],
            )
        except:
            pass

        # anotate the control signals
        for cname in control_names:
            try:

                power = int(np.floor(control_signals[cname][0] + 0.5))
                xt, yt = text_x[cname], text_y[cname]
                plt.annotate(
                    f"{cname} = {power:3d}",
                    (0, 0),
                    xytext=(xt, yt),
                    xycoords="data",
                    fontsize=20,
                    textcoords="data",
                    ha="left",
                    va="top",
                    color=text_colors[cname],
                )
            except:
                pass

        plt.title(f"{ftime:0.1f} h", fontsize=20)
        plt.xlabel("radius [mm]", fontsize=20)
        plt.ylabel("pixel signal", fontsize=20)
        plt.xlim(0, max_dist + 1)
        plt.ylim(0, max_signal + 10)
        plt.legend(fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()

        # store it as a figure
        if ofname is not None:

            # add the frame number to the output filename, and the roi ID to the folder
            ofname_complete = ofname + f"{frame:03d}"
            subdirs_complete = subdirs + f"{roi.id:03}/"

            fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
            flua.save_figure(
                fig, ofname_complete, file_ext, subdirs_complete, **fig_kwargs
            )

        plt.close()  # free up the memmory


def border_roi(
    roi, wide=0.1, offset=0.1, dist_chan="dist_inv_mm", frames=[-1], time_key="W"
):
    """
    To display the dynamic border of input wide and offset that moves as colony growths

    Parameters
    ----------

    wide: numeric
        wide of the border band

    offset: numeric
        distance to skip from the border

    dist_chan: str
        channel with the distance information to be used.

    frames: list
        frame number to get

    time_key: str
        key of the time serie to use from rot.times

    Returns
    -------
    bands_roi: dict
        dictionary with the bands of the roi at each indicated frame.
        the frame numbers are the dictionary keys.
        the bands are 'unit8' arrays with the surface = 255, band = 127 and exterior = 0

    """

    # init the dictionary to store the bands rois
    bands_roi = dict()

    # get the indicated channel of the roi
    dist_roi = roi.data[dist_chan]

    # go for each indicated frame
    for frame in frames:

        # get the channel frame
        dist_frame = dist_roi[:, :, frame]

        # init the band frame
        band_frame = np.zeros(dist_frame.shape, dtype="uint8")

        # get the list of distances
        dists = roi.rms[frame][0]

        # get the band distance limits
        try:
            up_limit = dists.max() - np.abs(offset)

        except:
            up_limit = 0

        low_limit = up_limit - wide

        # Fill the interior of the colony
        interior = dist_frame <= dists.max()
        band_frame[interior] = 255

        # get the indices of the band
        band = (dist_frame >= low_limit) & (dist_frame <= up_limit)
        band_frame[band] = 127

        # assign to the arrays
        bands_roi[frame] = band_frame

    return bands_roi


def plot_im_circle(
    image,
    center,
    radius,
    title_text=None,
    color="r",
    fill=False,
    lw=2,
    ofname=None,
    fformat=".pdf",
    overwrite=False,
    **kwargs,
):
    """
    to display a circle in the given image
    center = [xc, yc], point coordinates of circle center.
    """

    plt.imshow(image)
    plt.colorbar()
    if title_text is not None:
        plt.title(title_text)

    if isinstance(radius, (int, float, np.number)) and not isinstance(radius, bool):

        circle = plt.Circle((center), radius, color=color, fill=fill, lw=lw, **kwargs)
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle)

    # save the image file if indicated
    if ofname is not None:

        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        flua.save_figure(fig, ofname, fformat, overwrite=overwrite, **fig_kwargs)

    plt.show()


def circle_crop(image, center, r, background=1):

    # crops the monochromatic image to the area defined by the input circle
    # the area outside the circle take the value of background parameter.
    # if background = 'mean', fill the background of the image with its mean value.

    N, M = image.shape

    # to fill the background with the mean image value.
    try:
        if background == "mean":
            background = image.mean()
    except:
        pass

    im_circle = background * np.ones((N, M))
    y = center[0]
    x = center[1]

    for n in range(N):
        for m in range(M):
            if ((n - x) ** 2 + (m - y) ** 2) <= (r**2):
                im_circle[n, m] = image[n, m]

    plt.figure(figsize=(12, 3))

    plt.subplot(121)
    plt.imshow(image)
    plt.colorbar()
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(im_circle)
    plt.colorbar()
    plt.title("Circle crop image")

    return im_circle


def rois_plt_fluo_dynam(
    rois, time_v, cv, filename="null", channels=CHANNELS, fformat="pdf", overwrite=False
):
    """
    Plot the total fluorescence of each colony over time

    Parameters
    ----------
    rois: dictionary
        the ROI image array data (is better to use circular ROIS, obtained with obtain_rois() function)

    time_v: vector
        the vector of real time values

    cv: vector
        contain the ID of the of colonies analysed

    filename: string
        filename with whom save the output image with fluorescence dynamics

    channels: dict
        Channels position and name information.
        Structured as {channel_position : channel_name}

    """
    nchans = len(channels.keys())

    fig, ax = plt.figure(1, nchans, figsize=(4 * nchans, 3))

    for i, chan_name in enumerate(channels.values()):

        for k in cv:
            ax[i].plot(time_v, rois[chan_name][k].sum(axis=(0, 1)))  # sum the value
            # plt.hold(True)

        ax[i].set_xlabel("Time [h]")
        ax[i].set_ylabel("Fluorescence intensity")
        ax[i].set_title(chan_name + " channel")

    if filename != "null":

        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        flua.save_figure(fig, filename, fformat, overwrite=overwrite, **fig_kwargs)

    plt.show()
    # plt.legend(['Colony %d'%i for i in range(len(A))])


# def blobs_plot(frame, elements, annotate = False, labels = False, filename='null',
#               fformat = 'pdf', overwrite = False):
#    '''
#    ####### Deprecated ####
#
#    Crea una figura con los elementos circulares graficados sobre el frame
#    ingresado.
#
#    frame: image monochromatic o multichromatic.
#    elements: elements identified with skfeat.blob_log. Each element is an
#            array as it [y_pos, x_pos, sigma]
#
#    anotate: boolean
#      if True add the element number to each circle
#
#    labels: list
#      list with the labels of each element.
#
#    filename: string
#        filename with whom save the output image+blobs+ID
#
#    '''
#    # creamos la figura y desplegamos la imagen
#    plt.figure(figsize=(8,8))
#
#    if frame.ndim == 2:
#        plt.imshow(frame, cmap='gray')
#    else:
#        plt.imshow(frame)
#
#    # Agregamos los circulos
#    for i in range(len(elements)):
#
#        y = elements[i,0]
#        x = elements[i,1]
#        r = 2**(1/2)*elements[i,2]
#
#        circle = plt.Circle((x, y), r, color='r', fill=False , lw=0.5)
#        # circle(x_pos, y_pos, radio)
#
#        if annotate:
#          # label the ID on each element
#          try:
#            label = labels[i]
#          except:
#            label = str(i)
#
#          plt.annotate(label, xy=(x, y), xytext=(-int(r/2), int(r/2)),
#                      textcoords='offset points', ha='right', va='bottom',
#                      color='white')
#
#        fig = plt.gcf()
#        ax = fig.gca()
#        ax.add_artist(circle)
#
#    if filename != 'null':
#
#        fig_kwargs = {'bbox_inches': 'tight', 'transparent': True, 'dpi': 300}
#        flua.save_figure(fig, filename, fformat, overwrite = overwrite, **fig_kwargs)
#
#    plt.show()


def tl_roi(
    rois,
    times,
    idx,
    frames,
    fname="null",
    radius="null",
    chan_sum=False,
    same_bar=True,
    gridsize=[0, 0],
    channels=CHANNELS,
    fformat="pdf",
    overwrite=False,
):
    """
    Save images of selected time steps on "times" vector, for a selected ROI (idx).
    This images can be used to make timelapse videos of isolated colonies.

    If you specify a gridsize of a proper size, then it display the ROI frames on the notebook

    Parameters
    ----------
    rois: dictionary
            RGB time-lapse image data of each rois, from obtain_rois()

    times: vector
        contain the experimental time vector

    idx: intr
            contain the ID of the of the selected colony

    frames: vector
        conitains the selected time frames

    fname: string
        the complete filename to save the images of ROIs
        e.g. fname=('rois/Col'+str(idx)+'_ROI_step%d.png')

    chan_sum: boolean
        True to perfom the sum of the three channels of the ROI.
        False to show the image original colors.

    gridsize: vector
        size of the subplot grid. if gridsize=[0,0] the figure will not be shown on the notebook.

    channels: dict
        Channels position and name information.
        Structured as {channel_position : channel_name}

    Returns
    -------
    Save the images of the selected frames of a ROI.

    """
    if type(idx) == int:  # Check that ID is only one colony
        if len(frames) > 0:  # Check time vector have some value

            w1 = rois[channels[0]][idx].shape[0]
            h1 = rois[channels[0]][idx].shape[1]

            if chan_sum == True:
                ROIa = flua.channels_sum(rois, [idx])  # sum the three channels
                ROI = ROIa[idx][:, :, :]
                mx = np.max(ROI[:, :, :])
            else:
                # Reconstruct an image file for each time
                ROI = np.zeros((w1, h1, 3))

            # make the plot of each frame and save it
            roi = {}
            for i in frames:

                plt.figure(figsize=(8, 8))

                if chan_sum == True:  # Plot the ROI os sum with a colorbar
                    roi = ROI[:, :, i]
                    if same_bar == True:
                        plt.imshow(roi, interpolation="none", vmin=0, vmax=mx)
                    else:
                        plt.imshow(roi)
                    plt.colorbar()
                    plt.xticks([])
                    plt.yticks([])

                else:  # Plot the ROI original image
                    ROI[:, :, 0] = rois[channels[0]][idx][:, :, i]  # RED layer
                    ROI[:, :, 1] = rois[channels[1]][idx][:, :, i]  # GREEN layer
                    ROI[:, :, 2] = rois[channels[2]][idx][:, :, i]  # BlUE layer
                    roi[i] = ROI.astype("uint8")
                    plt.imshow(roi[i])
                    plt.xticks([])
                    plt.yticks([])

                if radius != "null":
                    cx = int(np.floor((w1 - 1) / 2 + 0.5))
                    cy = int(np.floor((h1 - 1) / 2 + 0.5))
                    circle = plt.Circle(
                        (cx, cy), radius[i], color="r", fill=False, lw=2
                    )
                    fig = plt.gcf()
                    ax = fig.gca()
                    ax.add_artist(circle)
                    # ax.axes.get_xaxis().set_visible(False)
                    # ax.axes.get_yaxis().set_visible(False)

                if fname != "null":
                    ofname = f"{fname}_t{times[i]:03d}"
                    fig_kwargs = {
                        "bbox_inches": "tight",
                        "transparent": True,
                        "dpi": 300,
                    }
                    flua.save_figure(
                        fig, ofname, fformat, overwrite=overwrite, **fig_kwargs
                    )

                plt.show()

            # display the plots in the notebook
            n = gridsize[0]
            m = gridsize[1]
            if (
                n and m > 0
            ):  # make n or m equal to zero to not display the figure in the notebook.
                if (n * m) < (len(frames)):
                    print(
                        "the subplot grid is smaller than the number of plots. Increase x or y, and try again"
                    )
                else:
                    plt.figure(figsize=(4 * m, 4 * n))
                    count = 1
                    for i in frames:
                        plt.subplot(n, m, count)

                        if chan_sum == True:
                            roi = ROI[:, :, i]
                            if same_bar == True:
                                plt.imshow(roi, interpolation="none", vmin=0, vmax=mx)
                            else:
                                plt.imshow(roi)
                            plt.colorbar()

                        else:

                            plt.imshow(roi[i])

                        if radius != "null":
                            cx = int(np.floor((w1 - 1) / 2 + 0.5))
                            cy = int(np.floor((h1 - 1) / 2 + 0.5))
                            circle = plt.Circle(
                                (cx, cy), radius[i], color="r", fill=False, lw=2
                            )
                            fig = plt.gcf()
                            ax = fig.gca()
                            ax.add_artist(circle)

                        plt.title(str(times[i]) + " Hours")
                        count += 1
        else:
            print("ERROR: Time vector have to be of lenght higher than zero")
    else:
        print("ERROR: use an integer value for the colony ID")


def logplot_radius(r, cv, t, filename="null", fformat="pdf", overwrite=False):
    """
    Plot the log of the square of the radius for each colony

    Parameters
    ----------
        r: dictionary
            colony radius at each time step of each colony at each time step (obtained with frame_colony_size() function)

        cv: vector
            colonies ID vector to plot

        T: vector
            the vector of real time values
        filename: string
            filename to save the plot generated
    """
    fig = plt.figure()

    for i in cv:
        R = r[i]
        plt.plot(t, np.log(R * R), ".")
        # plt.hold(True)
        plt.xlabel("Time [h]")
        plt.ylabel("log(Radius^2) [pixels]")
        plt.title("Colony radius")

    if filename != "null":

        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        flua.save_figure(fig, filename, fformat, overwrite=overwrite, **fig_kwargs)

    plt.show()


def plot_radius(
    r, cv, t, col_label=True, filename="null", fformat="pdf", overwrite=False
):
    """
    Plot the radius for each colony at each time step

    Parameters
    ----------
        r: dictionary
            colony radius at each time step of each colony (obtained with frame_colony_size() function)

        cv: vector
            colonies ID vector to plot

        t: vector
            the vector of real time values

        col_label: boolen
            to define if include or not the colony labels in the plot

        filename: string
            filename to save the plot generated
    """
    fig = plt.figure()

    for i in cv:
        R = r[i]

        if col_label == True:
            plt.plot(t, R, ".", label="colony " + str(i))
            plt.legend(loc="best")
        else:
            plt.plot(t, R, ".")

        plt.xlabel("Time [h]")
        plt.ylabel("Radius [pixels]")
        plt.title("Colony radius")

    if filename != "null":

        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        flua.save_figure(fig, filename, fformat, overwrite=overwrite, **fig_kwargs)

    plt.show()


def ROI_radius(
    rois,
    idx,
    frame=-1,
    r="null",
    filename="null",
    transect=False,
    plt_circle=False,
    fformat="pdf",
    overwrite=False,
):
    """
    Plot the colony radius estimate overlayed on an kymograph image slice

    Parameters
    ----------

        rois: dictionary
            ROI image of each colony (obtained with obtain_rois() function)

        idx: int
            id of the colony to check

        r:
            colony radius at each time step of the selected colony
            (obtained with frame_colony_radius() function)

        frame: int
            the frame to see

        filename: string
            filename to save the plot generated

        transect: boolean
            To indicate of include or not the middle transect line

        plt_circle: boolean
            To indicate of include or not the circle line
    """
    # get the roi
    roi = rois[idx]

    # get the roi dimentions and center rounded uo values
    h = roi.shape[0]  # roi.hw[0]  #rw
    w = roi.shape[1]  # roi.hw[1]  #cl

    # try:
    #    center = roi.get_center(rounded = True)
    # except:
    #    center = flua.round_up(roi.center)

    cy = flua.round_up(h / 2)  # = center[0]
    cx = flua.round_up(w / 2)  # = center[1]

    plt.figure()
    fig = plt.gcf()
    ax = fig.gca()
    ax.imshow(roi[:, :, frame], interpolation="none", cmap="gray")

    if transect == True:
        rect = Rectangle((0, cy), w, 0, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

    if plt_circle == True:
        circle = plt.Circle((cx, cy), r[idx][frame], color="w", fill=False, lw=0.5)
        ax.add_artist(circle)

    if filename != "null":

        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        flua.save_figure(fig, filename, fformat, overwrite=overwrite, **fig_kwargs)

    plt.show()


def check_radius(
    rois,
    idx,
    t,
    r_fit="null",
    r_dots="null",
    filename="null",
    transect=False,
    fformat="pdf",
    overwrite=False,
):
    """
    Plot the colony radius estimate overlayed on an kymograph image slice

    Parameters
    ----------
        r_fit: vector
            colony fited radius at each time step of the selected colony
            (obtained from a model)

        r_dots:
            colony radius at each time step of the selected colony
            (obtained with frame_colony_radius() function)

        rois: dictionary
            ROI image of each colony (obtained with obtain_rois() function)

        idx: int
            id of the colony to check

        t: vector
            the vector of real time values

        filename: string
            filename to save the plot generated

        transect: boolean
            To indicate of include or not the middle transect line

        overwrite: bool
            If False, you will be asked prior to overwrite the file
            If True, the files will be overwrittem directly.
    """

    # get the roi
    roi = rois[idx]

    # get the roi dimentions and center rounded uo values
    h = roi.shape[0]  # roi.hw[0]  #rw
    w = roi.shape[1]  # roi.hw[1]  #cl

    # try:
    #    center = roi.get_center(rounded = True)
    # except:
    #    center = flua.round_up(roi.center)

    cy = flua.round_up(h / 2)  # = center[0]
    cx = flua.round_up(w / 2)  # = center[1]

    # w,h,_ = rois[idx].shape
    fig, ax = plt.subplots(figsize=(18, 7))

    # use the y-middle transect
    im = ax.imshow(rois[idx][cy, :, :], interpolation="none")  # ,cmap='gray')
    fig.colorbar(im, fraction=0.04)

    if transect == True:
        rect2 = Rectangle(
            (0, cy), len(t) - 1, 0, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect2)

    if r_fit != "null":

        plt.plot(-r_fit + cy, "r-")  # quizas h/2 directamente para tener float(?)
        plt.plot(r_fit + cy, "r-")

    if r_dots != "null":

        plt.plot(
            -r_dots + cy, "rx", ms=9
        )  # quizas h/2 directamente para tener float(?)
        plt.plot(r_dots + cy, "rx", ms=9)

    step = int(len(t) * 0.1)
    plt.xticks(range(0, len(t), step), t[0:-1:step].astype(int))
    plt.xlabel("Time")
    plt.ylabel("y-axis position")
    plt.title("Colony " + str(idx))

    if filename != "null":

        fig_kwargs = {"bbox_inches": "tight", "transparent": True, "dpi": 300}
        flua.save_figure(fig, filename, fformat, overwrite=overwrite, **fig_kwargs)


def rois_last_frame_2chan_plt(rois_data, channel_x, channel_y, serie_name):
    """
    Sum all the pixel values for channel_x and channel_y (e.g.channel G and
    channel R) separately for the last frame of each ROI and make a plot where
    channel_x sum value is on the X axis and channel_y sum value is on the Y
    axis.
    Each dot correspond to one ROI values and represent the ratio of the two
    channels for that colony.

    Parameters
    ----------
        rois_data : dictionary
            RGB time-lapse image data of each ROI, obtained with obtain_rois()

        channel_x: string
            channel name (e.g. 'G') to be on the x axis

        channel_y: string
            channel name (e.g. 'R') to be on the y axis

        serie_name: string
            name of the the data serie in analysis (used as title of the plot)
    """

    # variable inicialization
    chanx = np.zeros((len(rois_data[channel_x]), 1))
    chany = np.zeros((len(rois_data[channel_x]), 1))

    # perform the sum of selected channels for the last frame of each ROI
    for i in range(len(rois_data[channel_x])):
        chanx[i] = rois_data[channel_x][i][:, :, -1].sum(axis=(0, 1))
        chany[i] = rois_data[channel_y][i][:, :, -1].sum(axis=(0, 1))

    # size the plot dimentions
    axisMax = np.max([np.max(chanx), np.max(chany)])
    axisMin = np.min([np.min(chanx), np.min(chany)])
    # print(axisMax,axisMin)

    # plt.figure(figsize=(8,8))
    plt.plot(chanx, chany, "bo")
    plt.title(serie_name)
    plt.xlabel(channel_x + " Channel")
    plt.ylabel(channel_y + " Channel")
    plt.axis([axisMin, axisMax, axisMin, axisMax])
    return (chanx, chany)


def plt_lin_fit(x_min, x_max, l_fit, color):
    """
    Make a plot of an already linear fit, being posible to define the function
    evaluation limits (x independient variable limits) and the color of the
    dots and fitted line.

    Parameters
    ----------
        x_min : int
            RGB time-lapse image data of each ROI, obtained with obtain_rois()

        x_max: int
            channel name (e.g. 'G') to be on the x axis

        l_fit: vector
            linear fitted parameters. Obtained with linear_fit function

        color: char
            char of the color to be used on the dots and line (e.g. 'r')
    """
    p = np.poly1d(l_fit)
    xp = np.linspace(x_min, x_max, 2)
    plt.plot(xp, p(xp), color + "-")
    # End

    ## function template
    # def plot_contours_size(ofname = None, **kwargs):
    """
    to plot the contour size of each given contour asociated to the given
    threhsolds.
    size in the sense of the pixel count in the contour.
    
    Parameters
    ----------
    contours: 
    
    ofname: string
        if given, the figure is stored under that name. More options
        with **kwargs  (see flua.save_fig())
    
    Return
    ------

    
    """
    """ 
    #init the 
    
    
    # make the figure
    fig, ax = plt.subplots(figsize = (8, 4))
    
    ax.plot(thresholds, cont_lengths, color = 'r', marker = '.')
    
    ax.set_xlabel('Threshold value [px]')
    ax.set_ylabel('Contour size [px]')
    ax.set_title(im_title)
    
    # save the figure file if indicated
    if type(ofname) is str:
        ofname += ''        
        fig_kwargs = {'bbox_inches': 'tight', 'transparent': True, 'dpi': 300}
        fig_kwargs.update(kwargs)  # join with the input functions kwargs
        flua.save_figure(fig, ofname, **fig_kwargs)
    
    return(cont_lengths)
    """
