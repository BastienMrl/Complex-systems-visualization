import { Vec3 } from "./ext/glMatrix/index.js";
import OBJFile from "./ext/objFileParser/OBJFile.js";
import { ShaderLocation } from "./shaderUtils.js";
// provides access to gl constants
const gl = WebGL2RenderingContext;
export class PlanMesh {
    _context;
    _vertPositions;
    _vertNormals;
    _vertUV;
    _vertIndices;
    _vao;
    constructor(context, size) {
        this._context = context;
        this._vao = this._context.createVertexArray();
        this._vertIndices = new Uint32Array([0, 1, 2, 0, 2, 3]);
        let scale = size / 2;
        this._vertNormals = new Float32Array(4 * 3).fill(0.);
        for (let i = 0; i < 4; i++) {
            this._vertNormals[i * 3 + 1] = 1;
        }
        this._vertPositions = new Float32Array(4 * 3).fill(0.);
        this._vertPositions[0] = -scale;
        this._vertPositions[2] = scale;
        this._vertPositions[3] = scale;
        this._vertPositions[5] = scale;
        this._vertPositions[6] = +scale;
        this._vertPositions[8] = -scale;
        this._vertPositions[9] = -scale;
        this._vertPositions[11] = -scale;
        this._vertUV = new Float32Array(4 * 2);
        this._vertUV[0] = 0;
        this._vertUV[1] = 1;
        this._vertUV[2] = 1;
        this._vertUV[3] = 1;
        this._vertUV[4] = 1;
        this._vertUV[5] = 0;
        this._vertUV[6] = 0;
        this._vertUV[7] = 0;
        this.initDrawVAO();
    }
    initDrawVAO() {
        this._context.bindVertexArray(this._vao);
        //positions
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ARRAY_BUFFER, this._vertPositions, gl.STATIC_DRAW);
        this._context.vertexAttribPointer(ShaderLocation.POS, 3, gl.FLOAT, false, 0, 0);
        this._context.enableVertexAttribArray(ShaderLocation.POS);
        // normals
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ARRAY_BUFFER, this._vertNormals, gl.STATIC_DRAW);
        this._context.vertexAttribPointer(ShaderLocation.NORMAL, 3, gl.FLOAT, false, 0, 0);
        this._context.enableVertexAttribArray(ShaderLocation.NORMAL);
        // uvs
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ARRAY_BUFFER, this._vertUV, gl.STATIC_DRAW);
        this._context.vertexAttribPointer(ShaderLocation.UV, 2, gl.FLOAT, false, 0, 0);
        this._context.enableVertexAttribArray(ShaderLocation.UV);
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
        this._context.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ELEMENT_ARRAY_BUFFER, this._vertIndices, gl.STATIC_DRAW);
        this._context.bindVertexArray(null);
    }
    draw() {
        this._context.bindVertexArray(this._vao);
        this._context.drawElements(gl.TRIANGLES, this._vertIndices.length, gl.UNSIGNED_INT, 0);
        this._context.bindVertexArray(null);
    }
}
export class MultipleMeshInstances {
    _context;
    _nbInstances;
    _aabb = new Float32Array(6);
    _localAabb = new Float32Array(6);
    _vertPositions;
    _vertNormals;
    _vertIndices;
    _uvBuffer;
    // TODO: defined by the user hardware
    _nbStates = 1;
    _vao;
    _mouseOverBuffer;
    constructor(context, values) {
        this._context = context;
        this._nbInstances = values.nbElements;
        this.updateAABB();
        this._vao = this._context.createVertexArray();
        this._uvBuffer = new InstanceAttribBuffer(context);
        let uvs = new Int32Array(values.nbElements * 2);
        let width = Math.ceil(Math.sqrt(values.nbElements));
        for (let i = 0; i < values.nbElements; ++i) {
            let u = Math.floor(i / width);
            let v = i % width;
            uvs[i * 2] = u;
            uvs[i * 2 + 1] = v;
        }
        this._uvBuffer.initialize(uvs);
    }
    // getters
    get vertPositions() {
        return this._vertPositions;
    }
    get vertNormals() {
        return this._vertNormals;
    }
    get aabb() {
        return this._aabb;
    }
    get nbRow() {
        return Math.sqrt(this._nbInstances);
    }
    get nbCol() {
        return Math.sqrt(this._nbInstances);
    }
    get nbInstances() {
        return this._nbInstances;
    }
    get localAabb() {
        return this._localAabb;
    }
    updateAABB() {
        let row = Math.sqrt(this._nbInstances);
        let offset = (row - 1) / 2.;
        this._aabb[0] = -offset;
        this._aabb[1] = offset;
        this._aabb[4] = -offset;
        this._aabb[5] = offset;
        this._aabb[2] = -2.;
        this._aabb[3] = 2.;
    }
    initDrawVAO() {
        this._context.bindVertexArray(this._vao);
        // positions
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ARRAY_BUFFER, new Float32Array(this._vertPositions), gl.STATIC_DRAW);
        this._context.vertexAttribPointer(ShaderLocation.POS, 3, gl.FLOAT, false, 0, 0);
        this._context.enableVertexAttribArray(0);
        // normals
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ARRAY_BUFFER, new Float32Array(this._vertNormals), gl.STATIC_DRAW);
        this._context.vertexAttribPointer(ShaderLocation.NORMAL, 3, gl.FLOAT, false, 0, 0);
        this._context.enableVertexAttribArray(1);
        // uvs
        this._uvBuffer.bindAttribs(ShaderLocation.UVS, 1, 2, gl.INT, false, 0);
        // mouse over
        this._mouseOverBuffer = this._context.createBuffer();
        const arr = new Float32Array(this._nbInstances).fill(0.);
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._mouseOverBuffer);
        this._context.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
        this._context.vertexAttribPointer(ShaderLocation.SELECTED, 1, gl.FLOAT, false, 0, 0);
        this._context.vertexAttribDivisor(ShaderLocation.SELECTED, 1);
        this._context.enableVertexAttribArray(ShaderLocation.SELECTED);
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
        this._context.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._context.createBuffer());
        this._context.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(this._vertIndices), gl.STATIC_DRAW);
        this._context.bindVertexArray(null);
    }
    extendLocalAabb(vertex) {
        let aabb = this._localAabb;
        let x = vertex[0];
        let y = vertex[1];
        let z = vertex[2];
        aabb[0] = aabb[0] < x ? aabb[0] : x;
        aabb[1] = aabb[1] > x ? aabb[1] : x;
        aabb[2] = aabb[2] < y ? aabb[2] : y;
        aabb[3] = aabb[3] > y ? aabb[3] : y;
        aabb[4] = aabb[4] < z ? aabb[4] : z;
        aabb[5] = aabb[5] > z ? aabb[5] : z;
    }
    updateMouseOverBuffer(indices) {
        const arr = new Float32Array(this._nbInstances).fill(0.);
        if (indices != null)
            indices.forEach((e) => {
                arr[e] = 1.;
            });
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._mouseOverBuffer);
        this._context.bufferSubData(gl.ARRAY_BUFFER, 0, arr);
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
    }
    draw() {
        this._context.bindVertexArray(this._vao);
        this._context.drawElementsInstanced(gl.TRIANGLES, this._vertIndices.length, gl.UNSIGNED_INT, 0, this._nbInstances);
        this._context.bindVertexArray(null);
    }
    async loadMesh(src) {
        this._localAabb = new Float32Array(6).fill(0);
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
            const v0 = element.vertices[0].vertexIndex - 1;
            let e1 = Vec3.create();
            let e2 = Vec3.create();
            Vec3.sub(e1, vertices[element.vertices[1].vertexIndex - 1], vertices[v0]);
            Vec3.sub(e2, vertices[element.vertices[2].vertexIndex - 1], vertices[v0]);
            let n = Vec3.create();
            Vec3.cross(n, e1, e2);
            for (let i = 1; i < element.vertices.length - 1; i++) {
                const v1 = element.vertices[i].vertexIndex - 1;
                const v2 = element.vertices[i + 1].vertexIndex - 1;
                vertIndices.push(v0, v1, v2);
            }
            for (let i = 0; i < element.vertices.length; i++) {
                const v = element.vertices[i].vertexIndex - 1;
                normals[v].add(n);
                nbFaces[v] += 1;
            }
        });
        for (let i = 0; i < vertices.length; ++i) {
            normals[i].scale(1. / nbFaces[i]);
            normals[i].normalize();
            this.extendLocalAabb(vertices[i]);
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
        let byteLength = 4;
        switch (type) {
            case gl.FLOAT:
                byteLength = 4;
                break;
            case gl.INT:
                byteLength = 4;
                break;
        }
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._bufferT0);
        for (let i = 0; i < nbLocations; i++) {
            let offset = i * size * byteLength;
            switch (type) {
                case gl.FLOAT:
                    this._context.vertexAttribPointer(location, size, type, normalized, stride, offset);
                    break;
                case gl.INT:
                    this._context.vertexAttribIPointer(location, size, type, stride, offset);
                    break;
            }
            this._context.vertexAttribDivisor(location, 1);
            this._context.enableVertexAttribArray(location);
            location++;
        }
        this._context.bindBuffer(gl.ARRAY_BUFFER, this._bufferT1);
        for (let i = 0; i < nbLocations; i++) {
            let offset = i * size * byteLength;
            switch (type) {
                case gl.FLOAT:
                    this._context.vertexAttribPointer(location, size, type, normalized, stride, offset);
                    break;
                case gl.INT:
                    this._context.vertexAttribIPointer(location, size, type, stride, offset);
                    break;
            }
            this._context.vertexAttribDivisor(location, 1);
            this._context.enableVertexAttribArray(location);
            location++;
        }
        this._context.bindBuffer(gl.ARRAY_BUFFER, null);
    }
}
