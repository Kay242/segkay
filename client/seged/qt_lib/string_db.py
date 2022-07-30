class QStringDB:

    btn_prev_image_tip = None
    btn_next_image_tip = None
    btn_delete_image_tip = None
    labeled_tip = None
    btn_view_lock_center_tip = None
    btn_view_xseg_mask_tip = None
    btn_view_xseg_overlay_mask_tip = None
    btn_view_baked_mask_tip = None
    btn_poly_color_blue_tip = None
    btn_poly_color_green_tip = None
    btn_poly_color_red_tip = None
    btn_pt_edit_mode_tip = None
    btn_delete_poly_tip = None
    btn_redo_pt_tip = None
    btn_undo_pt_tip = None
    btn_poly_type_exclude_tip = None
    btn_poly_type_include_tip = None

    @staticmethod
    def initialize():
        QStringDB.btn_poly_color_red_tip = 'Poly color scheme red'

        QStringDB.btn_poly_color_green_tip = 'Poly color scheme green'

        QStringDB.btn_poly_color_blue_tip = 'Poly color scheme blue'

        QStringDB.btn_view_baked_mask_tip = 'View baked mask'

        QStringDB.btn_view_xseg_mask_tip = 'View trained XSeg mask'

        QStringDB.btn_view_xseg_overlay_mask_tip = 'View trained XSeg mask overlay face'

        QStringDB.btn_poly_type_include_tip = 'Poly include mode'

        QStringDB.btn_poly_type_exclude_tip = 'Poly exclude mode'

        QStringDB.btn_undo_pt_tip = 'Undo point'

        QStringDB.btn_redo_pt_tip = 'Redo point'

        QStringDB.btn_delete_poly_tip = 'Delete poly'

        QStringDB.btn_pt_edit_mode_tip = 'Add/delete point mode ( HOLD CTRL )'

        QStringDB.btn_view_lock_center_tip = 'Lock cursor at the center ( HOLD SHIFT )'

        QStringDB.btn_prev_image_tip = 'Save and Prev image\nHold SHIFT : accelerate\nHold CTRL : skip non masked\n'
        QStringDB.btn_next_image_tip = 'Save and Next image\nHold SHIFT : accelerate\nHold CTRL : skip non masked\n'

        QStringDB.btn_delete_image_tip = 'Move to _trash and Next image\n'

        QStringDB.loading_tip = 'Loading'

        QStringDB.labeled_tip = 'labeled'
