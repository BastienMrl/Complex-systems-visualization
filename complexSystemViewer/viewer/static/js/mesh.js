import { Vec3 } from "./ext/glMatrix/index.js";
import OBJFile from "./ext/objFileParser/OBJFile.js";
// provides access to gl constants
const gl = WebGL2RenderingContext;
const posLoc = 0;
const normalLoc = 1;
const uvLoc = 2;
const translationLoc = 3;
const stateLoc = 5;
const selectionLoc = 10;
const idLoc = 1;
export class MultipleMeshInstances {
    _context;
    _nbInstances;
    _vertPositions;
    _vertNormals;
    _vertUVs;
    _vertIndices;
    _translationBuffer;
    _stateBuffer;
    _vao;
    _selectionVao;
    _mouseOverBuffer;
    constructor(context, values) {
        this._context = context;
        this._nbInstances = values.nbElements;
        this._vao = this._context.createVertexArray();
        this._selectionVao = this._context.createVertexArray();
        this._translationBuffer = new InstanceAttribBuffer(context);
        this._translationBuffer.initialize(values.translations);
        this._stateBuffer = new InstanceAttribBuffer(context);
        this._stateBuffer.initialize(values.states);
    }
    // getters
    get vertPositions() {
        return this._vertPositions;
    }
    get vertNormals() {
        return this._vertNormals;
    }
    // private methods
    getIndexFromCoords(i, j, nbCol) {
        return nbCol * i + j;
    }
    updataMouseOverBuffer(idx) {
        let arr = new Float32Array(this._nbInstances).fill(0.);
        if (idx != null)
            arr[idx] = 1.;
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._mouseOverBuffer);
        this._context.bufferSubData(gl.ARRAY_BUFFER, 0, arr);
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
    }
    initSelectionVAO() {
        this._context.bindVertexArray(this._selectionVao);
        // positions
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ARRAY_BUFFER, new Float32Array(this._vertPositions), gl.STATIC_DRAW);
        this._context.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
        this._context.enableVertexAttribArray(posLoc);
        // id
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        let ids = new Float32Array(this._nbInstances * 4);
        for (let i = 0; i < this._nbInstances; i++) {
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
        this._context.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(this._vertIndices), gl.STATIC_DRAW);
        this._context.bindVertexArray(null);
    }
    initDrawVAO() {
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
        // translation
        this._translationBuffer.bindAttribs(translationLoc, 1, 3, gl.FLOAT, false, 0);
        // states
        this._stateBuffer.bindAttribs(stateLoc, 1, 1, gl.FLOAT, false, 0);
        // mouse over
        this._mouseOverBuffer = this._context.createBuffer();
        const arr = new Float32Array(this._nbInstances).fill(0.);
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._mouseOverBuffer);
        this._context.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
        this._context.vertexAttribPointer(selectionLoc, 1, gl.FLOAT, false, 0, 0);
        this._context.vertexAttribDivisor(selectionLoc, 1);
        this._context.enableVertexAttribArray(selectionLoc);
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
        this._context.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(this._vertIndices), gl.STATIC_DRAW);
        this._context.bindVertexArray(null);
    }
    updateStates(values) {
        this._translationBuffer.updateAttribs(values.translations);
        this._stateBuffer.updateAttribs(values.states);
    }
    setMouseOver(idx) {
        this.updataMouseOverBuffer(idx);
    }
    draw() {
        this._context.bindVertexArray(this._vao);
        this._context.drawElementsInstanced(gl.TRIANGLES, this._vertIndices.length, gl.UNSIGNED_INT, 0, this._nbInstances);
        this._context.bindVertexArray(null);
    }
    drawSelection() {
        this._context.bindVertexArray(this._selectionVao);
        this._context.drawElementsInstanced(gl.TRIANGLES, this._vertIndices.length, gl.UNSIGNED_INT, 0, this._nbInstances);
        this._context.bindVertexArray(null);
    }
    async loadMesh(src) {
        const response = await fetch(src);
        const text = await response.text();
        const objFile = new OBJFile(text);
        const output = objFile.parse();
        const vertIndices = [];
        const vertices = new Array(output.models[0].vertices.length);
        output.models[0].vertices.forEach((e, idx) => {
            vertices[idx] = Vec3.fromValues(e.x, e.y, e.z);
        });
        const normals = new Array(vertices.length);
        for (let i = 0; i < normals.length; ++i) {
            normals[i] = Vec3.fromValues(0, 0, 0);
        }
        const nbFaces = new Array(vertices.length).fill(0);
        output.models[0].faces.forEach((element) => {
            for (let i = 1; i < element.vertices.length - 1; i++) {
                const v0 = element.vertices[0].vertexIndex - 1;
                const v1 = element.vertices[i].vertexIndex - 1;
                const v2 = element.vertices[i + 1].vertexIndex - 1;
                vertIndices.push(v0, v1, v2);
                let e1 = Vec3.create();
                let e2 = Vec3.create();
                Vec3.sub(e1, vertices[v1], vertices[v0]);
                Vec3.sub(e2, vertices[v2], vertices[v0]);
                let n = Vec3.create();
                Vec3.cross(n, e1, e2);
                normals[v0].add(n);
                normals[v1].add(n);
                normals[v2].add(n);
                nbFaces[v0] += 1;
                nbFaces[v1] += 1;
                nbFaces[v2] += 1;
            }
        });
        for (let i = 0; i < vertices.length; ++i) {
            normals[i].scale(1. / nbFaces[i]);
            normals[i].normalize();
        }
        this._vertPositions = new Float32Array(vertices.length * 3);
        for (let i = 0; i < vertices.length; i++) {
            this._vertPositions[i * 3] = vertices[i].x;
            this._vertPositions[i * 3 + 1] = vertices[i].y;
            this._vertPositions[i * 3 + 2] = vertices[i].z;
        }
        this._vertNormals = new Float32Array(normals.length * 3);
        for (let i = 0; i < normals.length; i++) {
            this._vertNormals[i * 3] = normals[i].x;
            this._vertNormals[i * 3 + 1] = normals[i].y;
            this._vertNormals[i * 3 + 2] = normals[i].z;
        }
        this._vertIndices = new Float32Array(vertIndices);
        this.initSelectionVAO();
        this.initDrawVAO();
    }
}
class InstanceAttribBuffer {
    _context;
    _bufferT0;
    _bufferT1;
    constructor(context) {
        this._context = context;
        this._bufferT0 = this._context.createBuffer();
        this._bufferT1 = this._context.createBuffer();
    }
    initialize(data) {
        console.log("initialization data = ", data);
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._bufferT0);
        this._context.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._bufferT1);
        this._context.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
    }
    updateAttribs(data) {
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._bufferT1);
        this._context.bindBuffer(gl.COPY_WRITE_BUFFER, this._bufferT0);
        this._context.copyBufferSubData(gl.ARRAY_BUFFER, gl.COPY_WRITE_BUFFER, 0, 0, data.byteLength);
        this._context.bufferSubData(gl.ARRAY_BUFFER, 0, data);
        this._context.bindBuffer(gl.COPY_WRITE_BUFFER, null);
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
    }
    bindAttribs(location, nbLocations, size, type, normalized, stride) {
        // assumes that type == gl.FLOAT
        const byteLength = 4;
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._bufferT0);
        for (let i = 0; i < nbLocations; i++) {
            let offset = i * size * byteLength;
            this._context.vertexAttribPointer(location, size, type, normalized, stride, offset);
            this._context.vertexAttribDivisor(location, 1);
            this._context.enableVertexAttribArray(location);
            location++;
        }
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._bufferT1);
        for (let i = 0; i < nbLocations; i++) {
            let offset = i * size * byteLength;
            this._context.vertexAttribPointer(location, size, type, normalized, stride, offset);
            this._context.vertexAttribDivisor(location, 1);
            this._context.enableVertexAttribArray(location);
            location++;
        }
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
    }
}
