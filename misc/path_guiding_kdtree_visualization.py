import path_guiding_kdtree_loader

import vtk
import sys
import os
import csv
from pprint import pprint, pformat
from matplotlib import pyplot
import numpy as np
import itertools
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
    ctf.SetRange(0., 1.)
    #ctf.Build()
    #return ctf
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


def generate_log_ctf(maxval):
    f = vtk.vtkDiscretizableColorTransferFunction()
    #f.DiscretizeOn()
    #f.SetColorSpaceToDiverging()
    f.SetNumberOfValues(256)
    f.AddRGBPoint(maxval*1.e1 , 1.0, 1.0, 1.0)
    f.AddRGBPoint(maxval*1.e0 , 0.9, 0.9, 0.9)
    f.AddRGBPoint(maxval*1.e-1, 0.6, 0.6, 0.0)
    f.AddRGBPoint(maxval*1.e-2, 0.9, 0.0, 0.0)
    f.AddRGBPoint(maxval*1.e-3, 0.6, 0.0, 0.6)
    f.AddRGBPoint(maxval*1.e-4, 0.0, 0.0, 0.9)
    f.AddRGBPoint(maxval*1.e-5, 0.0, 0.3, 0.3)
    f.AddRGBPoint(maxval*1.e-6, 0.1, 0.1, 0.1)
    f.Build()
    return f


lut = generate_lut()


# def pdf_image(ax, gmm):
#     t = np.linspace(-1.,1., 100)
#     x,y = np.meshgrid(t,t)
#     coord_list = np.vstack((x.ravel(), y.ravel())).T
#     pdf_vals = gmm.pdf(coord_list)
#     pdf_vals = np.reshape(pdf_vals, (t.size,t.size))
#     return ax.imshow(pdf_vals[::-1,::], extent=(-1.,1,-1.,1.))


class CellDisplay(object):
    def __init__(self, samplefile_pattern, main_renderer):
        #self.fig, self.axes = pyplot.subplots(1, 2, figsize = (10, 5))
        self.samplefile_pattern = samplefile_pattern
        self.main_renderer = main_renderer

    def display(self, cell1):
        print ("--- Cell {} ---".format(cell1.id))
        print (cell1)
        print (" ++ Learned means ++")
        print (cell1.mixture_learned.means)
        print (" ++ Learned weights ++")
        print (cell1.mixture_learned.weights)

        if not self.samplefile_pattern:
            return

        pos, dir, weight = path_guiding_kdtree_loader.load_sample_file(self.samplefile_pattern.format(cell1.id))
        weight /= weight.max()

        # fig, axes = self.fig, self.axes
        # for ax in axes:
        #     ax.clear()
        # fig.suptitle(f"#samples = {cell1.num_points}")
        # if cell1.val is not None:
        #     w = np.average(cell1.val, axis = 1) + 1.e-9
        #     norm = np.average(w)
        #     w /= norm
        #     axes[0].scatter(*cell1.proj.T, c = w, s = w, alpha = 0.2, edgecolor = 'none')
        #     axes[0].add_artist(pyplot.Circle((0, 0), 1, fill = False, color = 'k'))
        # pdf_image(axes[0], cell1.mixture_learned)
        # axes[0].set(title = 'learned')
        # pdf_image(axes[1], cell1.mixture_sampled)
        # axes[1].set(title = 'sampled')
        # pyplot.draw()
        # pyplot.show(block = False)

        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for i, p, d, w in zip(itertools.count(), pos, dir, weight):
            p = cell1.center
            rmin = np.average(cell1.stddev)*0.5
            pts.InsertNextPoint(p+d*rmin)
            pts.InsertNextPoint(p+d*(rmin+w*rmin))
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0,i*2)
            line.GetPointIds().SetId(1,i*2+1)
            lines.InsertNextCell(line)
        linesPolyData = vtk.vtkPolyData()
        linesPolyData.SetPoints(pts)
        linesPolyData.SetLines(lines)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(linesPolyData)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self.main_renderer.AddActor(actor)


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


def make_oriented_cube_actor_from_cell_data(cd, color):
    trafo = vtk.vtkTransform()
    m = np.zeros((4,4))
    # The cube has unit length, i.e. "radius 0.5". Hence, in contrast to
    # the rendering code, I don't need the factor 1/2 here.
    m[:3,:3] = 3.*cd.frame
    m[3,3] = 1
    m[:3,3] = cd.center
    trafo.SetMatrix(m.ravel())

    cube = vtk.vtkCubeSource()
    cube.Update()
    a = make_cube_actor(cube, color)
    a.SetUserTransform(trafo)

    return a



def make_cube_from_cell_box(cd):
    cube = vtk.vtkCubeSource()
    min_, max_ = cd.box.T
    cube.SetBounds(min_[0], max_[0], min_[1], max_[1], min_[2], max_[2])
    # cube.SetBounds(*cd.box.ravel())
    # cube.SetCenter(*cd.center)
    # cube.SetXLength(cd.stddev[0]*2)
    # cube.SetYLength(cd.stddev[1]*2)
    # cube.SetZLength(cd.stddev[2]*2)
    cube.Update()
    return cube


def generate_colors(polydata, cd):
    # The default radius is 0.5!
    pts = 2.*numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    # Color by pdf
    
    #vals = cd.mixture_learned.pdf(pts)
    #vals *= cd.incident_flux_learned
    # vals /= vals.max()

    # Color by incident radiance
    vals = np.ones(pts.shape[0]) * cd.incident_flux_learned

    return vals


def make_sphere_poly_data(cd):
    size = np.average(cd.stddev*3./2.)
    source = vtk.vtkSphereSource()
    source.SetThetaResolution(64)
    source.SetPhiResolution(32)
    source.Update()
    vals = generate_colors(source.GetOutput(), cd)
    source.SetCenter(*cd.center)
    source.SetRadius(size)
    source.Update()
    polydata = source.GetOutput()
    polydata.GetPointData().SetScalars(numpy_support.numpy_to_vtk(vals.astype(np.float32)))
    return polydata, vals.max()


def make_the_all_spheres_poly_data(datas):
    appendFilter = vtk.vtkAppendPolyData()
    maxval = 0.
    for pd, vals in datas:
        appendFilter.AddInputData(pd)
        maxval = max(maxval, vals)
    appendFilter.Update()
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(appendFilter.GetOutput())
    mapper.SetLookupTable(generate_log_ctf(maxval))
    mapper.SetColorModeToMapScalars()
    #mapper.SetScalarRange(0., maxval)
    #print ("maxval=",maxval)
    # Actor.
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


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

        if picker.GetActor() in self.actormap:
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
                #s = pformat(self.actormap[self.NewPickedActor])
                #self.textactor.SetInput(s)
                #print (s)
                self.display.display(self.actormap[self.NewPickedActor])

        self.OnLeftButtonDown()
        return


def filter_cell(cd):
    # if cd.num_points <= 1000:
    #     continue
    #if cd.id != 515:
    #    return False
    if np.any(np.abs(cd.center - np.array([-0.01, 0.4, -0.55])) > 0.3):
       return False
    return True

    
if __name__ == '__main__':
    filename = sys.argv[1]
    samplefile_pattern = sys.argv[2] if len(sys.argv)>2 else None

    assert(os.path.isfile(filename))

    actormap = {}
    tree, celldata = path_guiding_kdtree_loader.convert_data(
        path_guiding_kdtree_loader.read_records(filename),
        load_samples = False,
        build_boxes = False)

    # for i, cd in enumerate(celldata):
    #     cd['index'] = i

    ren = vtk.vtkRenderer()

    display = CellDisplay(samplefile_pattern, ren)

    colors = vtk.vtkNamedColors()

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

    #print (np.average([cd.incident_flux_learned for cd in celldata if filter_cell(cd)], axis=0))

    spheredatas = []
    for cd in celldata:
        if not filter_cell(cd):
            continue

        spheredatas.append(make_sphere_poly_data(cd))
        
        # pd, maxval = make_sphere_poly_data(cd)
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputData(pd)
        # mapper.SetLookupTable(lut)
        # mapper.SetColorModeToMapScalars()
        # mapper.SetScalarRange(0., 2.)
        # actor = vtk.vtkActor()
        # actor.SetMapper(mapper)
        # ren.AddActor(actor)

        # a = make_cube_actor(
        #     make_cube_from_cell_data(cd),
        #     colors.GetColor3d("Banana")
        # )
        #actormap[a] = cd
        #ren.AddActor(a)
        if 1:
            # a = make_cube_actor(
            #     make_cube_from_cell_data(cd),
            #     colors.GetColor3d("Gray")
            # )
            a = make_oriented_cube_actor_from_cell_data(cd, colors.GetColor3d("Gray"))
            actormap[a] = cd
            a.GetProperty().SetRepresentationToWireframe()
            a.GetProperty().LightingOff()
            ren.AddActor(a)
    a = make_the_all_spheres_poly_data(spheredatas)
    ren.AddActor(a)
    
    if os.path.isfile("/tmp/scene.obj") and 0:
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

    axes = vtk.vtkAxesActor()
    ren.AddActor(axes)

    ren.ResetCamera()
    ren.GetActiveCamera().Azimuth(30)
    ren.GetActiveCamera().Elevation(30)
    ren.ResetCameraClippingRange()
    ren.SetBackground(colors.GetColor3d("Black"))

    # Enable user interface interactor.
    iren.Initialize()
    renWin.Render()
    iren.Start()