"""Keep diagnostics JSON serializable and benchmark-safe."""
import json

def flatten(verification,recovery):
    return {'verification_status':verification.get('status','REVIEW'),
            'verification_reasons':json.dumps(verification.get('reasons',[])),
            'residual_ocr_boxes':verification.get('residual_ocr_boxes',0),
            'changed_outside_mask':verification.get('changed_outside_mask',0),
            'mask_boundary_gradient':verification.get('boundary_gradient',0),
            'recovery_attempted':recovery.get('attempted',False),
            'recovery_accepted':recovery.get('accepted',False)}
