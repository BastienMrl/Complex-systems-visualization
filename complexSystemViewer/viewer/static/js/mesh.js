import {vec3, mat4} from "./glMatrix/esm/index.js"

class Mesh{
    gl;

    #vertPositions;
    #vertNormals;
    #vertUVs;
    #nbFaces

    #modelMatrix;
    #vao;

    constructor(gl){
        this.gl = gl;

        this.#modelMatrix = mat4.create();
        mat4.identity(this.#modelMatrix);

        this.#vao = gl.createVertexArray();
    }

    rotate(rad, axis){
        mat4.rotate(this.#modelMatrix, this.#modelMatrix, rad, axis);
    }

    getWorldMatrix(){
        return this.#modelMatrix;
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



    getVertPositions() {
        return this.#vertPositions;
    }

    getVertNormals(){
        return this.#vertNormals;
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
        
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
        this.gl.bindVertexArray(null);
    }

    draw(){
        this.gl.bindVertexArray(this.#vao);
        this.gl.drawArrays(this.gl.TRIANGLES, 0, this.#nbFaces);
        this.gl.bindVertexArray(null);
    }
}

export {Mesh}