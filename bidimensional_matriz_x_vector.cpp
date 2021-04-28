/*
 ============================================================================
 Name        : bidimensional_matriz_x_vector.cpp
 Author      : Jose Saldaña Mercado
 Version     :
 Copyright   : GNU Open Souce and Free license
 Description : Multiplicacion de Matrix por Vector.
    Multiplica un vector por una matriz, repartiendo la matriz en submatrices cuadradas que procesa cada proceso.

 Build: mpicxx bidimensional_matriz_x_vector.cpp -o bi_mxv
 Run: mpirun --oversubscribe -np 4 bi_mxv
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
    long    *A, // Matriz a multiplicar
            *x, // Vector que vamos a multiplicar
            *y, // Vector donde almacenamos el resultado
            *subMatriz, // La submatriz que almacena localmente un proceso
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

    int raizP = sqrt(numeroProcesadores);

    if (n % raizP != 0) {
        if (idProceso == 0) {
            cout << "Uso: n debe ser multiplo de sqrt(numeroProcesadores)" << endl;
        }
        MPI_Finalize();
        return (0);
    }

    int tam = n / raizP; // Dimensión de las submatrices
    int nElem = tam * tam; // Numero de elementos que procesa cada procesador
    int filaP, columnaP; // indice de cada proceso dentro de la submatriz
    A = new long [nElem]; // Reservamos los elementos de cada proceso
    x = new long [n]; // El vector sera del mismo tamaño que una fila de la matriz

    MPI_Datatype MPI_BLOQUE; // Tipo de dato para reordenar la matriz y poder enviar submatrices como elementos consecutivos

    // Solo el proceso 0 ejecuta el siguiente bloque
    if (idProceso == 0) {
        long *auxiliar = new long[n * n];
        // Reservamos especio para el resultado
        y = new long [n];

        // Rellenamos 'auxiliar' y 'x' con valores aleatorios
        cout << "Inicio carga de datos........" << endl;
        srand(time(0));
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = 0; j < n; j++) {
                auxiliar[i * n + j] = rand() % 1000;
            }
            x[i] = rand() % 100;
        }
        cout << "........Fin carga de datos" << endl;

        if (n < 24) {
            cout << "La matriz y el vector generados son " << endl;
            for (unsigned int i = 0; i < n; i++) {
                for (unsigned int j = 0; j < n; j++) {
                    if (j == 0) cout << "[";
                    cout << auxiliar[i * n + j];
                    if (j == n - 1) cout << "]";
                    else cout << "  ";
                }
                cout << "\t  [" << x[i] << "]" << endl;
            }
            cout << "\n";
        }

        // Reservamos espacio para la comprobacion
        comprueba = new long [n];
        // Realizamos el algoritmo secuencial para comprobar
        cout << "Inicio algoritmo secuencial........" << endl;
	    tSecuencialIni = clock();
        // Lo calculamos de forma secuencial
        for (unsigned int i = 0; i < n; i++) {
            comprueba[i] = 0;
            for (unsigned int j = 0; j < n; j++) {
                comprueba[i] += auxiliar[i * n + j] * x[j];
            }
        }
	    tSecuencialFin = clock();
        cout << "........Fin algoritmo secuencial" << endl;
        tSecuencial = (tSecuencialFin - tSecuencialIni) / CLOCKS_PER_SEC;
        // Calculamos un solo valor para mostrar por pantalla si n grande
        for (unsigned int i = 0; i < n; i++) {
            compruebaSum += comprueba[i];
        }

        // Reordenamos matriz A ------------------------------------
        // Defino el tipo bloque cuadrado
        MPI_Type_vector (tam, tam, n, MPI_LONG, &MPI_BLOQUE);
        MPI_Type_commit (&MPI_BLOQUE);

        A = new long [n * n];

        cout << "NumeroP: " << numeroProcesadores << ", raizP: " << raizP << ", tam: " << tam << ", n: " << n << ", nElem: " << nElem << endl;
        
        int comienzo, posicion;
        for (int i = 0, posicion = 0; i < numeroProcesadores; i++) {
            // Calculo la posicion de comienzo de cada submatriz
            filaP = i / raizP;
            columnaP = i % raizP;
            comienzo = (columnaP * tam) + (filaP * tam * tam * raizP);
            MPI_Pack(&auxiliar[comienzo], 1, MPI_BLOQUE, A, sizeof(long) * n * n, &posicion, MPI_COMM_WORLD);
        }
        // Libero memoria de matriz auxiliar
        delete [] auxiliar;
        MPI_Type_free (&MPI_BLOQUE);
        // ---------------------------------------------------------
        if (n < 24) {
            cout << "La matriz reordenada es " << endl;
            for (unsigned int i = 0; i < n; i++) {
                for (unsigned int j = 0; j < n; j++) {
                    if (j == 0) cout << "[";
                    cout << A[i * n + j];
                    if (j == n - 1) cout << "]";
                    else cout << "  ";
                }
                cout << endl;
            }
            cout << "\n";
        }

    } // Termina el trozo de codigo que ejecuta solo 0

    // Reservamos espacio para la fila local de cada proceso
    subMatriz = new long [nElem];

    // Creamos comunicadores que seran necesarios:
    MPI_Comm filas, columnas, diagonal; // nuevos comunicadores

    filaP = idProceso / raizP;
    MPI_Comm_split(MPI_COMM_WORLD, // a partir del comunicador global.
        filaP, // los de la misma fila entraran en el mismo comunicador
        idProceso, // indica el orden de asignacion de rango dentro de los nuevos comunicadores
        &filas); // Referencia al nuevo comunicador creado.

    columnaP = idProceso % raizP;
    MPI_Comm_split(MPI_COMM_WORLD, columnaP, idProceso, &columnas);

    int inDiagonal = MPI_UNDEFINED;
    if (filaP == columnaP) {
        inDiagonal = 1;
    }
    MPI_Comm_split(MPI_COMM_WORLD, inDiagonal, idProceso, &diagonal);

    // Hacemos una barrera para asegurar que todas los procesos comiencen la ejecucion
    // a la vez, para tener mejor control del tiempo empleado
    MPI_Barrier(MPI_COMM_WORLD);
    // Inicio de medicion de tiempo
    tInicio = MPI_Wtime();

    long *subFinal = new long [raizP];

    for (unsigned int i = 0; i < raizP; i++) {
        subFinal[i] = 0;
        for (unsigned int j = 0; j < n; j++) {
            // cout << "proc=" << idProceso << ", i=" << i << ", j=" << j << ", subfinal[" << i << "] += " << subMatriz[(i * n) + j] << " * " << x[j] << endl;
            subFinal[i] += subMatriz[(i * n) + j] * x[j];
        }
    }

    // Otra barrera para asegurar que todas ejecuten el siguiente trozo de c�digo lo
    // mas proximamente posible
    MPI_Barrier(MPI_COMM_WORLD);
    // fin de medicion de tiempo
    tFin = MPI_Wtime();

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
    delete [] subMatriz;

}

