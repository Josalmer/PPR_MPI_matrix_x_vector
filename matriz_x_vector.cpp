/*
 ============================================================================
 Name        : matriz_x_vector2.cpp
 Author      : Jose Saldaña Mercado
 Version     :
 Copyright   : GNU Open Souce and Free license
 Description : Multiplicacion de Matrix por Vector.
    Multiplica un vector por una matriz.

 Build: mpicxx matriz_x_vector.cpp -o mxv
 Run: mpirun --oversubscribe -np 4 mxv
 ============================================================================
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <cmath>

using namespace std;

int main(int argc, char * argv[]) {

    int numeroProcesadores,
            idProceso;
    long **A, // Matriz a multiplicar
            *x, // Vector que vamos a multiplicar
            *y, // Vector donde almacenamos el resultado
            *misFilas, // Las filas que almacena localmente un proceso
            *comprueba; // Guarda el resultado final (calculado secuencialmente), su valor
                        // debe ser igual al de 'y'

    long compruebaSum = 0; // Para mostrar resultado de comprobación para valores de n > 24
    long ySum = 0; // Para mostrar resultado de comprobación para valores de n > 24

    double tInicio, // Tiempo en el que comienza la ejecucion
            tFin, // Tiempo en el que acaba la ejecucion
            tSecuencialIni,
            tSecuencialFin,
            tSecuencial;

    int n;
    if (argc != 2) {
        if (idProceso == 0) {
            cout << "Uso: Pasar como parámetro dimensión de la matriz n" << endl;
        }
        return (0);
    }
    else {
        n = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numeroProcesadores);
    MPI_Comm_rank(MPI_COMM_WORLD, &idProceso);

    int nFilas = n / numeroProcesadores; // Numero de filas que procesa cada procesador
    int nElem = nFilas * n; // Numero de elementos que procesa cada procesador
    A = new long *[nElem]; // Reservamos las filas de la matriz
    x = new long [n]; // El vector sera del mismo tamaño que una fila de la matriz

    // Variables para n % numeroProcesadores != 0
    int filasUltimo = n - ((numeroProcesadores - 1) * nFilas);
    int *elementosPorProcesador = new int[numeroProcesadores];
    int *displenv = new int[numeroProcesadores];
    int *displrecv = new int[numeroProcesadores];

    // Solo el proceso 0 ejecuta el siguiente bloque
    if (idProceso == 0) {
        A[0] = new long [n * n];
        for (unsigned int i = 1; i < n; i++) {
            A[i] = A[i - 1] + n; // Para poder referenciar filas dentro de la matriz A, reservada en memoria como un vector largo
        }
        // Reservamos especio para el resultado
        y = new long [n];

        // Rellenamos 'A' y 'x' con valores aleatorios
        cout << "Inicio carga de datos........" << endl;
        srand(time(0));
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = 0; j < n; j++) {
                A[i][j] = rand() % 1000;
            }
            x[i] = rand() % 100;
        }
        cout << "........Fin carga de datos" << endl;

        if (n < 24) {
            cout << "La matriz y el vector generados son " << endl;
            for (unsigned int i = 0; i < n; i++) {
                for (unsigned int j = 0; j < n; j++) {
                    if (j == 0) cout << "[";
                    cout << A[i][j];
                    if (j == n - 1) cout << "]";
                    else cout << "  ";
                }
                cout << "\t  [" << x[i] << "]" << endl;
            }
            cout << "\n";
        }

        // Calculamos valores para n % numeroProcesadores != 0 -------------------
        cout << numeroProcesadores - 1 << " procesadores procesan " << nFilas << " filas y el ultimo procesa " << filasUltimo << " filas" << endl;

        // Filas de cada procesador
        for (int i = 0; i < numeroProcesadores -1; i++) {
            elementosPorProcesador[i] = nFilas * n;
        }
        elementosPorProcesador[numeroProcesadores - 1] = filasUltimo * n;
        cout << "Elementos que procesa cada procesador: [";
        for (int i = 0; i < numeroProcesadores; i++) {
            cout << " " << elementosPorProcesador[i];
        }
        cout << " ]" << endl;

        // Desplazamiento en vectores
        for (int i = 0; i < numeroProcesadores; i++) {
            displenv[i] = i * nFilas * n;
            displrecv[i] = i * nFilas;
        }
        cout << "Desplazamiento de envío para cada vector: [";
        for (int i = 0; i < numeroProcesadores; i++) {
            cout << " " << displenv[i];
        }
        cout << " ]" << endl;
        cout << "Desplazamiento de recepción para cada vector: [";
        for (int i = 0; i < numeroProcesadores; i++) {
            cout << " " << displrecv[i];
        }
        cout << " ]" << endl;

        // Reservamos espacio para la comprobacion
        comprueba = new long [n];
        // Realizamos el algoritmo secuencial para comprobar
        cout << "Inicio algoritmo secuencial........" << endl;
	    tSecuencialIni = clock();
        // Lo calculamos de forma secuencial
        for (unsigned int i = 0; i < n; i++) {
            comprueba[i] = 0;
            for (unsigned int j = 0; j < n; j++) {
                comprueba[i] += A[i][j] * x[j];
                compruebaSum += A[i][j] * x[j];
            }
        }
	    tSecuencialFin = clock();
        cout << "........Fin algoritmo secuencial" << endl;
        tSecuencial = (tSecuencialFin - tSecuencialIni) / CLOCKS_PER_SEC;
    } // Termina el trozo de codigo que ejecuta solo 0

    // Ajustamos nº de filas del ultimo procesador
    if (idProceso == (numeroProcesadores - 1)) {
        nFilas = filasUltimo;
        nElem = nFilas * n;
    }

    // Reservamos espacio para la fila local de cada proceso
    misFilas = new long [nElem];

    MPI_Scatterv(A[0], // Matriz que vamos a compartir
            elementosPorProcesador, // Numero de datos a compartir
            displenv, // Desplazamiento dentro de los datos a compartir
            MPI_LONG, // Tipo de dato a enviar
            misFilas, // Vector en el que almacenar los datos
            nElem, // Numero de datos a compartir
            MPI_LONG, // Tipo de dato a recibir
            0, // Proceso raiz que envia los datos
            MPI_COMM_WORLD); // Comunicador utilizado (En este caso, el global)

    // Compartimos el vector entre todas los procesos
    MPI_Bcast(x, // Dato a compartir
            n, // Numero de elementos que se van a enviar y recibir
            MPI_LONG, // Tipo de dato que se compartira
            0, // Proceso raiz que envia los datos
            MPI_COMM_WORLD); // Comunicador utilizado (En este caso, el global)


    // Hacemos una barrera para asegurar que todas los procesos comiencen la ejecucion
    // a la vez, para tener mejor control del tiempo empleado
    MPI_Barrier(MPI_COMM_WORLD);
    // Inicio de medicion de tiempo
    tInicio = MPI_Wtime();

    long *subFinal = new long [nFilas];

    for (unsigned int i = 0; i < nFilas; i++) {
        subFinal[i] = 0;
        for (unsigned int j = 0; j < n; j++) {
            // cout << "proc=" << idProceso << ", i=" << i << ", j=" << j << ", subfinal[" << i << "] += " << misFilas[(i * n) + j] << " * " << x[j] << endl;
            subFinal[i] += misFilas[(i * n) + j] * x[j];
        }
    }

    // Otra barrera para asegurar que todas ejecuten el siguiente trozo de c�digo lo
    // mas proximamente posible
    MPI_Barrier(MPI_COMM_WORLD);
    // fin de medicion de tiempo
    tFin = MPI_Wtime();

    // Recogemos los datos de la multiplicacion, por cada proceso sera un escalar
    // y se recoge en un vector, Gather se asegura de que la recolecci�n se haga
    // en el mismo orden en el que se hace el Scatter, con lo que cada escalar
    // acaba en su posicion correspondiente del vector.
    MPI_Gatherv(subFinal, // Dato que envia cada proceso
            nFilas, // Numero de elementos que se envian
            MPI_LONG, // Tipo del dato que se envia
            y, // Vector en el que se recolectan los datos
            elementosPorProcesador, // Numero de datos que se esperan recibir por cada proceso
            displrecv, // displs
            MPI_LONG, // Tipo del dato que se recibira
            0, // proceso que va a recibir los datos
            MPI_COMM_WORLD); // Canal de comunicacion (Comunicador Global)

    // Terminamos la ejecucion de los procesos, despues de esto solo existira
    // el proceso 0
    // Ojo! Esto no significa que los demas procesos no ejecuten el resto
    // de codigo despues de "Finalize", es conveniente asegurarnos con una
    // condicion si vamos a ejecutar mas codigo (Por ejemplo, con "if(rank==0)".
    MPI_Finalize();

    if (idProceso == 0) {

        unsigned int errores = 0;

        cout << "El resultado obtenido y el esperado son:" << endl;
        for (unsigned int i = 0; i < n; i++) {
            ySum += y[i];
            if (n < 24) {
                cout << "\t" << y[i] << "\t|\t" << comprueba[i] << endl;
            }
            if (comprueba[i] != y[i])
                errores++;
        }
        cout << "\tSUMA DE VECTORES: " << ySum << "\t|\t" << compruebaSum << endl;

        delete [] y;
        delete [] comprueba;
        delete [] A[0];

        if (errores) {
            cout << "Hubo " << errores << " errores." << endl;
        } else {
            cout << "No hubo errores" << endl;
            cout << "El tiempo paralelo ha sido " << tFin - tInicio << " segundos." << endl;
            cout << "El tiempo secuencial ha sido " << tSecuencial << " segundos." << endl;
            cout << "La ganancia ha sido " << tSecuencial/(tFin - tInicio) << endl;
        }

    }

    delete [] x;
    delete [] A;
    delete [] misFilas;

}

