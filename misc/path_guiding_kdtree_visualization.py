import path_guiding_kdtree_loader

import vtk
import sys
import os
from pprint import pprint, pformat
from matplotlib import pyplot
import numpy as np
from vtk.util import numpy_support

def generate_lut():
    colors = vtk.vtkNamedColors()
    # Colour transfer function.
    ctf = vtk.vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()
    p1 = [0.0] + list(colors.GetColor3d("MidnightBlue"))
    p2 = [1.0] + list(colors.GetColor3d("DarkOrange"))
    ctf.AddRGBPoint(*p1)
    ctf.AddRGBPoint(*p2)
    cc = list()
    for i in range(256):
        cc.append(ctf.GetColor(float(i) / 255.0))
    # Lookup table.
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfColors(256)
    for i, item in enumerate(cc):
        lut.SetTableValue(i, item[0], item[1], item[2], 1.0)
    lut.SetRange(0, 1)
    lut.Build()
    return lut

lut = generate_lut()


def pdf_image(ax, gmm):
    t = np.linspace(-1.,1., 100)
    x,y = np.meshgrid(t,t)
    coord_list = np.vstack((x.ravel(), y.ravel())).T
    pdf_vals = gmm.pdf(coord_list)
    pdf_vals = np.reshape(pdf_vals, (t.size,t.size))
    return ax.imshow(pdf_vals[::-1,::], extent=(-1.,1,-1.,1.))


class CellDisplay(object):
    def __init__(self):
        self.fig, self.axes = pyplot.subplots(1, 2, figsize = (10, 5))

    def display(self, cell1):
        fig, axes = self.fig, self.axes
        for ax in axes:
            ax.clear()
        fig.suptitle(f"#samples = {cell1.num_points}")
        # if cell1.val is not None:
        #     w = np.average(cell1.val, axis = 1) + 1.e-9
        #     norm = np.average(w)
        #     w /= norm
        #     axes[0].scatter(*cell1.proj.T, c = w, s = w, alpha = 0.2, edgecolor = 'none')
        #     axes[0].add_artist(pyplot.Circle((0, 0), 1, fill = False, color = 'k'))
        pdf_image(axes[0], cell1.mixture_learned)
        axes[0].set(title = 'learned')
        pdf_image(axes[1], cell1.mixture_sampled)
        axes[1].set(title = 'sampled')
        pyplot.draw()
        pyplot.show(block = False)


def make_cube_from_cell_data(cd):
    center = cd.center
    size   = cd.stddev
    cube = vtk.vtkCubeSource()
    cube.SetXLength(size[0])
    cube.SetYLength(size[1])
    cube.SetZLength(size[2])
    cube.SetCenter(*center)
    cube.Update()
    return cube


def generate_colors(polydata, cd):
    pts = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    #print(pts)
    vals = cd.mixture_learned.pdf(pts)
    vtkdata = numpy_support.numpy_to_vtk(vals.astype(np.float32))
    polydata.GetPointData().SetScalars(vtkdata)
    return vals


def make_sphere_from_cell_data(cd):
    center = cd.center
    size = np.average(cd.stddev)*0.5*10
    source = vtk.vtkSphereSource()
    source.SetThetaResolution(32)
    source.SetPhiResolution(16)
    source.Update()
    #
    polydata = source.GetOutput()
    vals = generate_colors(polydata, cd)
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetLookupTable(lut)
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarRange(0., vals.max()) #max(1./np.pi*0.25, vals.max()))
    #print (vals.max())
    # Actor.
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetPosition(*center)
    actor.SetScale(size, size, size)
    return actor


def make_cube_from_cell_box(cd):
    cube = vtk.vtkCubeSource()
    cube.SetBounds(*cd.box.ravel())
    cube.Update()
    return cube


def make_cube_actor(cube, color):
    # mapper
    cubeMapper = vtk.vtkPolyDataMapper()
    cubeMapper.SetInputData(cube.GetOutput())
    # Actor.
    cubeActor = vtk.vtkActor()
    cubeActor.SetMapper(cubeMapper)
    cubeActor.GetProperty().SetColor(color)
    return cubeActor


class MouseInteractorHighLightActor(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, actormap, textactor, display):
        self.AddObserver("LeftButtonPressEvent", self.leftButtonPressEvent)
        self.LastPickedActor = None
        self.LastPickedProperty = vtk.vtkProperty()
        self.actormap = actormap
        self.textactor = textactor
        self.display = display

    def leftButtonPressEvent(self, obj, event):
        clickPos = self.GetInteractor().GetEventPosition()

        picker = vtk.vtkPropPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())

        # get the new
        self.NewPickedActor = picker.GetActor()

        # If something was selected
        if self.NewPickedActor:
            # If we picked something before, reset its property
            if self.LastPickedActor:
                self.LastPickedActor.GetProperty().DeepCopy(self.LastPickedProperty)

            # Save the property of the picked actor so that we can
            # restore it next time
            self.LastPickedProperty.DeepCopy(self.NewPickedActor.GetProperty())
            # Highlight the picked actor by changing its properties
            self.NewPickedActor.GetProperty().SetColor(1.0, 0.0, 0.0)
            self.NewPickedActor.GetProperty().SetDiffuse(1.0)
            self.NewPickedActor.GetProperty().SetSpecular(0.0)
            self.NewPickedActor.GetProperty().SetLineWidth(3.0)
            # save the last picked actor
            self.LastPickedActor = self.NewPickedActor
            #assert self.NewPickedActor in self.actormap

            if self.NewPickedActor in self.actormap:
                s = pformat(self.actormap[self.NewPickedActor])
                #self.textactor.SetInput(s)
                print (s)
                self.display.display(self.actormap[self.NewPickedActor])

        self.OnLeftButtonDown()
        return

    
if __name__ == '__main__':
    filename = sys.argv[1]
    assert(os.path.isfile(filename))

    actormap = {}
    tree, celldata = path_guiding_kdtree_loader.convert_data(
        path_guiding_kdtree_loader.read_records(filename),
        load_samples = False,
        build_boxes = True)
    for i, cd in enumerate(celldata):
        cd['index'] = i

    display = CellDisplay()

    colors = vtk.vtkNamedColors()

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.SetWindowName("Cube")
    renWin.AddRenderer(ren)

    textActor = vtk.vtkTextActor()
    #textActor.SetTextScaleModeToProp()
    textActor.SetDisplayPosition(90, 50)
    textActor.GetTextProperty().SetFontSize(18)
    textActor.GetTextProperty().SetFontFamilyToArial()
    textActor.SetInput("")
    ren.AddActor(textActor)

    # Create a renderwindowinteractor.
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    style = MouseInteractorHighLightActor(actormap = actormap, textactor = textActor, display = display)
    style.SetDefaultRenderer(ren)
    iren.SetInteractorStyle(style)

    for cd in celldata:
        # if cd.num_points <= 1000:
        #     continue
        a = make_sphere_from_cell_data(cd)
        # a = make_cube_actor(
        #     make_cube_from_cell_data(cd),
        #     colors.GetColor3d("Banana")
        # )
        #actormap[a] = cd
        ren.AddActor(a)
        if 0:
            a = make_cube_actor(
                make_cube_from_cell_box(cd),
                colors.GetColor3d("Gray")
            )
            actormap[a] = cd
            a.GetProperty().SetRepresentationToWireframe()
            a.GetProperty().LightingOff()
            ren.AddActor(a)
    
    if os.path.isfile("/tmp/scene.obj"):
        # Because of the number format
        os.environ['LC_NUMERIC']='en_US.UTF-8'
        importer = vtk.vtkOBJReader()
        importer.SetFileName("/tmp/scene.obj")
        importer.Update()
        pd = importer.GetOutput()

        # ID_TO_REMOVE = 2
        # group_ids = numpy_support.vtk_to_numpy(pd.GetCellData().GetArray("GroupIds")).astype(np.int32)
        # selected_ids, = np.where(group_ids != ID_TO_REMOVE)
        # print (selected_ids)

        # selectionNode = vtk.vtkSelectionNode()
        # selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        # selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        # selectionNode.SetSelectionList(numpy_support.numpy_to_vtk(selected_ids))

        # selection = vtk.vtkSelection()
        # selection.AddNode(selectionNode)

        # selection_filter = vtk.vtkExtractSelectedPolyDataIds()
        # selection_filter.SetInputConnection(0, importer.GetOutputPort())
        # selection_filter.SetInputData(1, selection)
        # selection_filter.Update()

        # pd = selection_filter.GetOutput()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        ren.AddActor(actor)

    ren.ResetCamera()
    ren.GetActiveCamera().Azimuth(30)
    ren.GetActiveCamera().Elevation(30)
    ren.ResetCameraClippingRange()
    ren.SetBackground(colors.GetColor3d("Black"))

    # Enable user interface interactor.
    iren.Initialize()
    renWin.Render()
    iren.Start()