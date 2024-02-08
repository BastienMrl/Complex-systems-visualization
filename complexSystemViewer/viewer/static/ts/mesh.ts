import { Vec3, Mat4 } from "./glMatrix/index.js";

// provides access to gl constants
const gl = WebGL2RenderingContext;

export class MultipleMeshInstances{
    
    private _context : WebGL2RenderingContext;
    private _nbInstances : number;

    private _vertPositions : Float32Array;
    private _vertNormals : Float32Array;
    private _vertUVs : Float32Array;
    private _nbFaces : number;

    private _modelMatrices : Float32Array;
    private _matrixBuffer : WebGLBuffer | null;

    private _colors : Float32Array;
    private _colorBuffer : WebGLBuffer | null;

    private _vao : WebGLVertexArrayObject | null;
    private _selectionVao : WebGLVertexArrayObject | null;
    private _mouseOverBuffer : WebGLBuffer | null;

    public constructor(context : WebGL2RenderingContext, nbInstances : number){
        this._context = context;
        this._nbInstances = nbInstances;

        let matrix : Mat4 = new Mat4();
        matrix.identity();

        this._modelMatrices = new Float32Array(nbInstances * 16);
        for(let i = 0; i < nbInstances * 16; i += 16){
            for(let j = 0; j < 16; j++){
                this._modelMatrices[i + j] = matrix[j];
            }
        }

        this._colors = new Float32Array(nbInstances * 3);
        this._colors[0] = 1.;
        this._colors[1] = 1.;
        this._colors[2] = 1.;
        for (let i = 3; i < nbInstances * 3; i += 3){
            this._colors[i] = 0;
            this._colors[i + 1] = 0.5;
            this._colors[i + 2] = 0.5;
        }
        
        this._vao = this._context.createVertexArray();
        this._selectionVao = this._context.createVertexArray();
    }

    // getters
    public get vertPositions() {
        return this._vertPositions;
    }

    public get vertNormals(){
        return this._vertNormals;
    }
    

    // private methods
    private getIndexFromCoords(i : number, j : number, nbCol : number) : number {
        return nbCol * i + j; 
    }

    private updateMatrixBuffer(){
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._matrixBuffer);
        this._context.bufferSubData(gl.ARRAY_BUFFER, 0, this._modelMatrices);
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    private updateColorBuffer(){
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._colorBuffer);
        this._context.bufferSubData(gl.ARRAY_BUFFER, 0, this._colors);
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    private updataMouseOverBuffer(idx : number | null){
        let arr = new Float32Array(this._nbInstances).fill(0.);
        if (idx != null)
            arr[idx] = 1.;
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._mouseOverBuffer);
        this._context.bufferSubData(gl.ARRAY_BUFFER, 0, arr);
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    private initSelectionVAO(){
        this._context.bindVertexArray(this._selectionVao);

        // positions
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ARRAY_BUFFER, new Float32Array(this._vertPositions), gl.STATIC_DRAW);
        this._context.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
        this._context.enableVertexAttribArray(0);

        // id
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        let ids = new Float32Array(this._nbInstances * 4);
        for (let i = 0; i < this._nbInstances; i++){
            ids[i * 4] = (((i + 1) & 0x000000FF) >> 0) / 0xFF;
            ids[i * 4 + 1] = (((i + 1) & 0x0000FF00) >> 8) / 0xFF;
            ids[i * 4 + 2] = (((i + 1) & 0x00FF0000) >> 16) / 0xFF;
            ids[i * 4 + 3] = (((i + 1) & 0xFF000000) >> 24) / 0xFF;
        }
        this._context.bufferData(gl.ARRAY_BUFFER, ids, gl.STATIC_DRAW);
        this._context.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);
        this._context.vertexAttribDivisor(1, 1);
        this._context.enableVertexAttribArray(1);

        // world matrices
        const matrixLoc = 2
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._matrixBuffer);
        this._context.bufferData(gl.ARRAY_BUFFER, this._modelMatrices, gl.DYNAMIC_DRAW);
        for (let i = 0; i < 4; i++){
            let location = matrixLoc + i;
            let offset = 16 * i;
            this._context.vertexAttribPointer(location, 4, gl.FLOAT, false, 4 * 16, offset);
            this._context.vertexAttribDivisor(location, 1);
            this._context.enableVertexAttribArray(location)
        }
        
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
        this._context.bindVertexArray(null);
    }

    private updateBuffersVAO(){
        this._context.bindVertexArray(this._vao);

        // positions
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ARRAY_BUFFER, new Float32Array(this._vertPositions), gl.STATIC_DRAW);
        this._context.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
        this._context.enableVertexAttribArray(0);

        // normals
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ARRAY_BUFFER, new Float32Array(this._vertNormals), gl.STATIC_DRAW);
        this._context.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
        this._context.enableVertexAttribArray(1);
        
        // colors
        this._colorBuffer = this._context.createBuffer();
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._colorBuffer);
        this._context.bufferData(gl.ARRAY_BUFFER, this._colors, gl.DYNAMIC_DRAW);
        this._context.vertexAttribPointer(3, 3, gl.FLOAT, false, 0, 0);
        this._context.vertexAttribDivisor(3, 1);
        this._context.enableVertexAttribArray(3);
        
        // mouse over
        this._mouseOverBuffer = this._context.createBuffer();
        const arr = new Float32Array(this._nbInstances).fill(0.);
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._mouseOverBuffer);
        this._context.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
        this._context.vertexAttribPointer(4, 1, gl.FLOAT, false, 0, 0);
        this._context.vertexAttribDivisor(4, 1);
        this._context.enableVertexAttribArray(4);
        
        // world matrices
        this._matrixBuffer = this._context.createBuffer();
        const matrixLoc = 5;
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._matrixBuffer);
        this._context.bufferData(gl.ARRAY_BUFFER, this._modelMatrices, gl.DYNAMIC_DRAW);
        for (let i = 0; i < 4; i++){
            let location = matrixLoc + i;
            let offset = 16 * i;
            this._context.vertexAttribPointer(location, 4, gl.FLOAT, false, 4 * 16, offset);
            this._context.vertexAttribDivisor(location, 1);
            this._context.enableVertexAttribArray(location)
        }
        
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
        this._context.bindVertexArray(null);
    }

    // public methods
    public applyGridLayout(firstPos : Vec3, nbRow : number, nbCol : number, offsetRow : Vec3, offsetCol : Vec3){
        let matrix = new Mat4().identity().translate(firstPos);
        for(let i = 0; i < nbRow; i++){
            let colMatrix = new Mat4().copy(matrix);
            for(let j = 0; j < nbCol; j++){
                for (let k = 0; k < 16; k++){
                    let index : number = this.getIndexFromCoords(i, j, nbCol) * 16 + k;
                    this._modelMatrices[index] = colMatrix[k];
                }
                colMatrix.translate(offsetCol);
            }
            matrix.translate(offsetRow);
        }
        this.updateMatrixBuffer()
    }

    public updateYpos(data : Array<number>){
        for (let i = 0; i < data.length; i++){
            this._modelMatrices[i * 16 + 13] = data[i] ? 1. : 0;
        }
        this.updateMatrixBuffer();
    }

    public updateColors(colors : Float32Array){
        this._colors = colors;
        this.updateColorBuffer();
    }

    public setColor(color : Vec3, idx : number){
        for (let i = 0; i < 3; ++i){
            this._colors[idx * 3 + i] = color[i]
        }
        this.updateColorBuffer();
    }

    public setMouseOver(idx : number | null){
        this.updataMouseOverBuffer(idx);
    }

    public draw(){
        this._context.bindVertexArray(this._vao);
        this._context.drawArraysInstanced(gl.TRIANGLES, 0, this._nbFaces, this._nbInstances);
        this._context.bindVertexArray(null);
    }

    public drawSelection(){
        this._context.bindVertexArray(this._selectionVao);
        this._context.drawArraysInstanced(gl.TRIANGLES, 0, this._nbFaces, this._nbInstances);
        this._context.bindVertexArray(null);
    }

    public loadCube(){
        this._vertPositions = new Float32Array([
            // Front face
            -1.0, -1.0,  1.0,
            1.0, -1.0,  1.0,
            1.0,  1.0,  1.0,
            
            -1.0, -1.0,  1.0,
            1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0,
            
            // Back face
            -1.0, -1.0, -1.0,
            -1.0,  1.0, -1.0,
            1.0,  1.0, -1.0,
            
            -1.0, -1.0, -1.0,
            1.0,  1.0, -1.0,
            1.0, -1.0, -1.0,
            
            // Top face
            -1.0,  1.0, -1.0,
            -1.0,  1.0,  1.0,
            1.0,  1.0,  1.0,

            -1.0,  1.0, -1.0,
            1.0,  1.0,  1.0,
            1.0,  1.0, -1.0,
        
            // Bottom face
            -1.0, -1.0, -1.0,
            1.0, -1.0, -1.0,
            1.0, -1.0,  1.0,

            -1.0, -1.0, -1.0,
            1.0, -1.0,  1.0,
            -1.0, -1.0,  1.0,
            
            // Right face
            1.0, -1.0, -1.0,
            1.0,  1.0, -1.0,
            1.0,  1.0,  1.0,

            1.0, -1.0, -1.0,
            1.0,  1.0,  1.0,
            1.0, -1.0,  1.0,
        
            // Left face
            -1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0,
            -1.0,  1.0,  1.0,

            -1.0, -1.0, -1.0,
            -1.0,  1.0,  1.0,
            -1.0,  1.0, -1.0
        ]);

        this._vertNormals = new Float32Array(this._vertPositions.length);
        let normals : Vec3[] = [
            Vec3.fromValues(0., 0, 1.),
            Vec3.fromValues(0., 0., -1.),
            Vec3.fromValues(0., 1, 0.),
            Vec3.fromValues(0., -1.0, 0.),
            Vec3.fromValues(1.0, 0., 0.),
            Vec3.fromValues(-1., 0, 0),
        ]
        let cpt : number = 0;
        normals.forEach(element => {
            for (let i = 0; i < 6; i++)
                for(let j = 0; j < 3; j++)
                    this._vertNormals[cpt++] = element[j];
        });

        this._nbFaces = this._vertPositions.length / 3;
        this.updateBuffersVAO();
        this.initSelectionVAO();
    }
}