import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix
import numpy as np
from mathutils.geometry import intersect_sphere_sphere_2d as sss

bl_info = {
    "name": "Rockhead Games snap menu extension",
    "author": "Fernando D'Andrea",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Edit Mode, Mesh > Snap",
    "description": "Add operations to relocate objects and sub-objects.",
    "category": "3D View"
}


def calculate_circumcenter_3d(p1, p2, p3):

    def barycentric_to_world(p1, p2, p3, u, v, w):

        return (u*p1 + v*p2 + w*p3) / (u + v + w)

    a = p3 - p2
    b = p1 - p3
    c = p2 - p1

    u = a.length_squared * b.dot(c)
    v = b.length_squared * c.dot(a)
    w = c.length_squared * b.dot(a)

    if (u + v + w) == 0:
        report({'WARNING'}, "The 3 points are colinear. Using average instead!")
        return (p1 + p2 + p3) / 3

    try:
        center = barycentric_to_world(p1, p2, p3, u, v, w)
    except ZeroDivisionError:
        return None
    except ValueError:
        return None

    return center


def calculate_circumcenter_4d(A, B, C, D):
    """
    Calculates the circumcenter of a sphere that touches four points in 3D space.
    Args:
        A, B, C, D (mathutils.Vector): Points in 3D space.
    Returns:
        circumcenter (mathutils.Vector): The circumcenter of the sphere.
    """

    # Convert to numpy arrays
    A, B, C, D = np.array(A), np.array(B), np.array(C), np.array(D)

    # Create the system matrix, with rows corresponding to the equations (Ax + By + Cz - D = 0)
    M = np.array([
        [2*(B[0]-A[0]), 2*(B[1]-A[1]), 2*(B[2]-A[2])],
        [2*(C[0]-B[0]), 2*(C[1]-B[1]), 2*(C[2]-B[2])],
        [2*(D[0]-C[0]), 2*(D[1]-C[1]), 2*(D[2]-C[2])]
    ])

    # Create the right-hand side vector
    RHS = np.array([
        B[0]**2 + B[1]**2 + B[2]**2 - A[0]**2 - A[1]**2 - A[2]**2,
        C[0]**2 + C[1]**2 + C[2]**2 - B[0]**2 - B[1]**2 - B[2]**2,
        D[0]**2 + D[1]**2 + D[2]**2 - C[0]**2 - C[1]**2 - C[2]**2
    ])

    # Solve the system of equations
    try:
        center = np.linalg.solve(M, RHS)
    except np.linalg.LinAlgError:
        return None

    return Vector(center)



class MoveToCircumcenterOperator(bpy.types.Operator):
    bl_idname = "mesh.cursor_to_circumcenter_operator"
    bl_label = "Cursor to Circumcenter"
    bl_icon = "MESH_CIRCLE"
    bl_description = "Move the cursor to the circumcenter of the circle/sphere defined by the other 3/4 selected entities"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        if obj is None:
            return False

        if obj.mode == 'EDIT':
            me = obj.data
            bm = bmesh.from_edit_mesh(me)
            selected_verts = [v for v in bm.verts if v.select]
            active_vert = bm.select_history.active

            # if len(selected_verts) != 5 or not isinstance(active_vert, bmesh.types.BMVert):
            if len(selected_verts) == 4 or len(selected_verts) == 3:
                return True

        if obj.mode == 'OBJECT':
            if len(context.selected_objects) == 4 or len(context.selected_objects) == 3:
                return True

        # check if object is armature and mode is pose
        if obj.type == 'ARMATURE':
            if obj.mode == 'POSE':
                #check if 3 or 4 bones are selected
                if len(context.selected_pose_bones) == 4 or len(context.selected_pose_bones) == 3:
                    return True

            if obj.mode == 'EDIT':
                #check if 3 or 4 bones are selected
                if len(context.selected_bones) == 4 or len(context.selected_bones) == 3:
                    return True

        return False

    def execute(self, context):

        obj = context.object

        #check if object contains vertices
        p = [];

        if obj.type == 'MESH' and obj.mode == 'EDIT':
            obj = bpy.context.edit_object
            me = obj.data
            bm = bmesh.from_edit_mesh(me)

            p = [obj.matrix_world @ v.co for v in bm.verts if v.select]

        # if obj.mode is object mode, get selected objects positions
        if obj.mode == 'OBJECT':
            p = [obj.location for obj in context.selected_objects]

        # if obj is armature and mode is pose, get selected bones positions
        if obj.type == 'ARMATURE':
            if obj.mode == 'POSE':
                p = [obj.matrix_world @ bone.head for bone in context.selected_pose_bones]
            if obj.mode == 'EDIT':
                p = [obj.matrix_world @ bone.head_local for bone in context.selected_bones]

        if len(p) == 3:
            circumcenter = calculate_circumcenter_3d(p[0], p[1], p[2])

        if len(p) == 4:
            circumcenter = calculate_circumcenter_4d(p[0], p[1], p[2], p[3])
            if circumcenter is None:
                circumcenter = calculate_circumcenter_3d(p[0], p[1], p[2])
                if circumcenter is not None:
                    self.report({'WARNING'}, "The 4 points are coplanar. Using 3D circumcenter instead!")

        if circumcenter is None:
            self.report({'ERROR'}, "Could not calculate circumcenter. Are the points colinear?")
            return {'CANCELLED'}

        # Display the distance between the circumcenter and the input points:
        for i in range(len(p)):
            self.report({'INFO'}, "Distance from circumcenter to point {}: {}".format(i, (circumcenter - p[i]).length))

        context.scene.cursor.location = circumcenter

        return {'FINISHED'}


class LookAtCursorOperator(bpy.types.Operator):
    bl_idname = "view3d.look_at_cursor_operator"
    bl_label = "Look at Cursor"
    bl_icon = "HIDE_OFF"
    bl_description = "Points the object to the 3D cursor"
    bl_options = {'REGISTER', 'UNDO'}

    axis: bpy.props.EnumProperty(
        name="Axis",
        description="Axis that will point to cursor",
        items=[
            ('X', "X", "X Axis"),
            ('Y', "Y", "Y Axis"),
            ('Z', "Z", "Z Axis"),
        ],
        default='Z',
    )

    reverse: bpy.props.BoolProperty(
        name = "Reverse",
        description = "Reverse the direction of the axis",
        default = False
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'axis')
        layout.prop(self, 'reverse')

    @classmethod
    def poll(cls, context):
        return len(context.selected_objects) > 0


    def execute(self, context):
        # obj = context.active_object
        cursor_location = context.scene.cursor.location

        for obj in context.selected_objects:

            direction = cursor_location - obj.location
            if self.reverse:
                direction = obj.location - cursor_location
            direction.normalize()

            try:
                if self.axis == 'X':
                    obj_rot = direction.to_track_quat('X', 'Z')
                elif self.axis == 'Y':
                    obj_rot = direction.to_track_quat('Y', 'Z')
                else:
                    obj_rot = direction.to_track_quat('Z', 'Y')
            except ValueError:
                self.report({'ERROR'}, "Could not point object to cursor. Are the object and cursor in the same location?")
                return {'CANCELLED'}

            obj.rotation_euler = obj_rot.to_euler()

        return {'FINISHED'}

def snap_menu_draw(self, context):
    layout = self.layout
    layout.separator()  # Add a separator
    layout.operator(LookAtCursorOperator.bl_idname, icon=LookAtCursorOperator.bl_icon);
    layout.operator(MoveToCircumcenterOperator.bl_idname, icon=MoveToCircumcenterOperator.bl_icon)

def register():
    bpy.utils.register_class(LookAtCursorOperator)
    bpy.utils.register_class(MoveToCircumcenterOperator)
    bpy.types.VIEW3D_MT_snap.append(snap_menu_draw)

def unregister():
    bpy.utils.unregister_class(LookAtCursorOperator)
    bpy.utils.unregister_class(MoveToCircumcenterOperator)
    bpy.types.VIEW3D_MT_snap.remove(snap_menu_draw)


