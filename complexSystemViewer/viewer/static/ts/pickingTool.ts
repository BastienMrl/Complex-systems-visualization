import { MultipleMeshInstances } from "./mesh";

export class PickingTool {
    private _meshes : MultipleMeshInstances;

    public setMeshes(meshes : MultipleMeshInstances){
        this._meshes = meshes;
    }
}