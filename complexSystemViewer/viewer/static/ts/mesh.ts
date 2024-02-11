import { Vec3, Mat4 } from "./glMatrix/index.js";

// provides access to gl constants
const gl = WebGL2RenderingContext;

const posLoc = 0;
const normalLoc = 1;
const uvLoc = 2;

const colorLoc = 3;
const translationLoc = 5;

const selectionLoc = 10;


const idLoc = 1;

export class MultipleMeshInstances{
    
    private _context : WebGL2RenderingContext;
    private _nbInstances : number;

    private _vertPositions : Float32Array;
    private _vertNormals : Float32Array;
    private _vertUVs : Float32Array;
    private _nbFaces : number;

    private _colorBuffer : InstanceAttribBuffer;
    private _colors : Float32Array;

    private _translationBuffer : InstanceAttribBuffer;
    private _translation : Float32Array;

    private _vao : WebGLVertexArrayObject | null;
    private _selectionVao : WebGLVertexArrayObject | null;
    private _mouseOverBuffer : WebGLBuffer | null;

    public constructor(context : WebGL2RenderingContext, nbInstances : number){
        this._context = context;
        this._nbInstances = nbInstances;

        this._translation = new Float32Array(nbInstances * 3);

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
        this._translationBuffer = new InstanceAttribBuffer(context);
        this._translationBuffer.initialize(this._translation);

        this._colorBuffer = new InstanceAttribBuffer(context);
        this._colorBuffer.initialize(this._colors);
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
        this._context.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
        this._context.enableVertexAttribArray(posLoc);

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
        this._context.vertexAttribPointer(idLoc, 4, gl.FLOAT, false, 0, 0);
        this._context.vertexAttribDivisor(idLoc, 1);
        this._context.enableVertexAttribArray(idLoc);

        // translation
        this._translationBuffer.bindAttribs(translationLoc, 1, 3, gl.FLOAT, false, 0);
        
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
        this._context.bindVertexArray(null);
    }

    private initDrawVAO(){
        this._context.bindVertexArray(this._vao);

        // positions
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ARRAY_BUFFER, new Float32Array(this._vertPositions), gl.STATIC_DRAW);
        this._context.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
        this._context.enableVertexAttribArray(0);

        // normals
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ARRAY_BUFFER, new Float32Array(this._vertNormals), gl.STATIC_DRAW);
        this._context.vertexAttribPointer(normalLoc, 3, gl.FLOAT, false, 0, 0);
        this._context.enableVertexAttribArray(1);
        
        // colors
        this._colorBuffer.bindAttribs(colorLoc, 1, 3, gl.FLOAT, false, 0);

        // translation
        this._translationBuffer.bindAttribs(translationLoc, 1, 3, gl.FLOAT, false, 0);
        
        // mouse over
        this._mouseOverBuffer = this._context.createBuffer();
        const arr = new Float32Array(this._nbInstances).fill(0.);
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._mouseOverBuffer);
        this._context.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
        this._context.vertexAttribPointer(selectionLoc, 1, gl.FLOAT, false, 0, 0);
        this._context.vertexAttribDivisor(selectionLoc, 1);
        this._context.enableVertexAttribArray(selectionLoc);
        
        
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
        this._context.bindVertexArray(null);
    }

    // public methods
    public applyGridLayout(firstPos : Vec3, nbRow : number, nbCol : number, offsetRow : Vec3, offsetCol : Vec3){
        let rowPos = firstPos;
        for(let i = 0; i < nbRow; i++){
            let colPos = new Vec3().copy(rowPos);
            for(let j = 0; j < nbCol; j++){
                for (let k = 0; k < 3; k++){
                    let index : number = this.getIndexFromCoords(i, j, nbCol) * 3 + k;
                    this._translation[index] = colPos[k];
                }
                colPos.add(offsetCol);
            }
            rowPos.add(offsetRow);
        }
        this._translationBuffer.updateAttribs(this._translation)
        this._colorBuffer.updateAttribs(this._colors);
    }

    public updateTranslations(translations : Float32Array){
        this._translationBuffer.updateAttribs(translations);
    }
    
    public updateColors(colors : Float32Array){
        this._colorBuffer.updateAttribs(colors);
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
        this.initDrawVAO();
        this.initSelectionVAO();
    }
}


class InstanceAttribBuffer{
    private _context : WebGL2RenderingContext;
    
    private _bufferT0 : WebGLBuffer;
    private _bufferT1 : WebGLBuffer;

    constructor(context : WebGL2RenderingContext){
        this._context = context;

        this._bufferT0 = this._context.createBuffer();
        this._bufferT1 = this._context.createBuffer();
    }

    public initialize(data : Float32Array){
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._bufferT0);
        this._context.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
        
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._bufferT1);
        this._context.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
        
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    public updateAttribs(data : Float32Array){
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._bufferT1);
        this._context.bindBuffer(gl.COPY_WRITE_BUFFER, this._bufferT0);

        this._context.copyBufferSubData(gl.ARRAY_BUFFER, gl.COPY_WRITE_BUFFER, 0, 0, data.byteLength);
        this._context.bufferSubData(gl.ARRAY_BUFFER, 0, data);

        this._context.bindBuffer(gl.COPY_WRITE_BUFFER, null);
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    public bindAttribs(location : number, nbLocations : number, size : number,
                        type : number, normalized : boolean, stride : number){
                            
        // assumes that type == gl.FLOAT
        const byteLength = 4;
        
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._bufferT0);
        for (let i = 0; i < nbLocations; i++){
            let offset = i * size * byteLength;
            this._context.vertexAttribPointer(location, size, type, normalized, stride, offset);
            this._context.vertexAttribDivisor(location, 1);
            this._context.enableVertexAttribArray(location);
            location++;
        }


        this._context.bindBuffer(gl.ARRAY_BUFFER, this._bufferT1);
        for (let i = 0; i < nbLocations; i++){
            let offset = i * size * byteLength;
            this._context.vertexAttribPointer(location, size, type, normalized, stride, offset);
            this._context.vertexAttribDivisor(location, 1);
            this._context.enableVertexAttribArray(location);
            location++;
        }


        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
    }
}