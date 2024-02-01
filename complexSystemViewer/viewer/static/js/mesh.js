import {vec3, mat4} from "./glMatrix/esm/index.js"

class MultipleMeshInstances{
    gl;
    nbInstances;

    #vertPositions;
    #vertNormals;
    #vertUVs;
    #nbFaces

    #modelMatrices;
    #matrixBuffer;

    #colors;
    #colorBuffer;

    #vao;
    #nbCol;

    constructor(gl, nbInstances){
        this.gl = gl;
        this.nbInstances = nbInstances;

        let matrix = mat4.create();
        mat4.identity(matrix);

        this.#modelMatrices = new Float32Array(nbInstances * 16);
        for(let i = 0; i < nbInstances * 16; i += 16){
            for(let j = 0; j < 16; j++){
                this.#modelMatrices[i + j] = matrix[j];
            }
        }

        this.#colors = new Float32Array(nbInstances * 3);
        this.#colors[0] = 0.5;
        this.#colors[1] = 0.5;
        this.#colors[2] = 0;
        for (let i = 3; i < nbInstances * 3; i += 3){
            this.#colors[i] = 0;
            this.#colors[i + 1] = 0.5;
            this.#colors[i + 2] = 0.5;
        }


        this.#vao = gl.createVertexArray();
    }

    loadCube(){
        this.#vertPositions = [
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
        ];

        this.#vertNormals = [];
        let normals = [
            vec3.fromValues(0., 0, -1.),
            vec3.fromValues(0., 0., 1.),
            vec3.fromValues(0., -1, 0.),
            vec3.fromValues(0., 1.0, 0.),
            vec3.fromValues(-1.0, 0., 0.),
            vec3.fromValues(1., 0, 0),
        ]
        normals.forEach(element => {
            for (let i = 0; i < 6; i++)
                for(let j = 0; j < 3; j++)
                    this.#vertNormals.push(element[j]);
        });

        this.#nbFaces = this.#vertPositions.length / 3;
        this.updateBuffersVAO();
    }

    #getIndexFromCoords(i, j, nbCol){
        return nbCol * i + j; 
    }

    applyGridLayout(firstPos, nbRow, nbCol, offsetRow, offsetCol){
        let output = vec3.create();
        
        let matrix = mat4.create();
        mat4.identity(matrix);
        mat4.translate(matrix, matrix, firstPos);
        for(let i = 0; i < nbRow; i++){
            let colMatrix = mat4.create();
            mat4.copy(colMatrix, matrix);
            for(let j = 0; j < nbCol; j++){
                for (let k = 0; k < 16; k++){
                    let index = this.#getIndexFromCoords(i, j, nbCol) * 16 + k;
                    this.#modelMatrices[index] = colMatrix[k];
                }
                mat4.translate(colMatrix, colMatrix, offsetCol);
            }
            mat4.translate(matrix, matrix, offsetRow);
        }
        this.#updateMatrixBuffer()
        this.#nbCol = nbCol
    }

    setColor(i, j, color){
        let index = this.#getIndexFromCoords(i, j, this.#nbCol);
        for (let k = 0; k < 3; k++ ){
            this.#colors[index * 3 + k] = color[k];
        }

    }


    getVertPositions() {
        return this.#vertPositions;
    }

    getVertNormals(){
        return this.#vertNormals;
    }

    #updateMatrixBuffer(){
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.#matrixBuffer);
        this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, this.#modelMatrices);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
    }

    updateColorBuffer(){
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.#colorBuffer);
        this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, this.#colors);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
    }

    updateBuffersVAO(){
        this.gl.bindVertexArray(this.#vao);

        // positions
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.gl.createBuffer());
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(this.getVertPositions()), this.gl.STATIC_DRAW);
        this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0);
        this.gl.enableVertexAttribArray(0);

        // normals
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.gl.createBuffer());
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(this.getVertNormals()), this.gl.STATIC_DRAW);
        this.gl.vertexAttribPointer(1, 3, this.gl.FLOAT, false, 0, 0);
        this.gl.enableVertexAttribArray(1);

        // colors
        this.#colorBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.#colorBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.#colors, this.gl.DYNAMIC_DRAW);
        this.gl.vertexAttribPointer(3, 3, this.gl.FLOAT, false, 0, 0);
        this.gl.vertexAttribDivisor(3, 1);
        this.gl.enableVertexAttribArray(3);

        // world matrices
        this.#matrixBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.#matrixBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.#modelMatrices, this.gl.DYNAMIC_DRAW);
        for (let i = 0; i < 4; i++){
            let location = 4 + i;
            let offset = 16 * i;
            this.gl.vertexAttribPointer(location, 4, this.gl.FLOAT, false, 4 * 16, offset);
            this.gl.vertexAttribDivisor(location, 1);
            this.gl.enableVertexAttribArray(location)
        }
        
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
        this.gl.bindVertexArray(null);
    }

    draw(){
        this.gl.bindVertexArray(this.#vao);
        this.gl.drawArraysInstanced(this.gl.TRIANGLES, 0, this.#nbFaces, this.nbInstances);
        this.gl.bindVertexArray(null);
    }
}

export {MultipleMeshInstances}