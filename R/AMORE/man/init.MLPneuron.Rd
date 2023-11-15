\name{init.MLPneuron}
\alias{init.MLPneuron}
\title{Neuron constructor.}
\description{Creates a neuron according to the structure established by the AMORE package standard.}
\usage{
init.MLPneuron(id, type, activation.function, output.links, output.aims, input.links, 
        weights, bias, method, method.dep.variables)
}
\arguments{
\item{id}{Numerical index of the neuron (so as to be refered in a network operation).}
\item{type}{Either hidden or ouput,  according to the layer the neuron belongs to.}
\item{activation.function}{The name of the characteristic function of the neuron. It can be "pureline", "tansig", "sigmoid" or even "custom" in case that the user wants to configure its own activation function accordingly defining f0 and f1.}
\item{output.links}{The id's of the neurons that accept the output value of this neuron as an input.}
\item{output.aims}{The location of the output of the neuron in the input set of the addressed neuron. Gives answer to: Is this output the first, the second, the third, ..., input at the addressed neuron?. Similarly for an output neuron: Is this output the first, the second, the third, ..., element of the output vector?}
\item{input.links}{The id's of the neurons whose outputs work as inputs for this neuron. Positive values represent that we take the outputs of other neurons as inputs. Negative values represent the coordinates of the input vector to be considered as inputs.}
\item{weights}{The multiplying factors of the input values.}
\item{bias}{The bias summed to the weighted sum of the inputs.}
\item{method}{Prefered training method. Currently it can be:
        \itemize{
        \item "ADAPTgd": Adaptative gradient descend.
        \item "ADAPTgdwm": Adaptative gradient descend with momentum.
        \item "BATCHgd": BATCH gradient descend.
        \item "BATCHgdwm": BATCH gradient descend with momentum.
        }
}

\item{method.dep.variables}{Variables used by the training methods:
        \itemize{
        \item ADAPTgd method:
                \itemize{
                        \item delta: Used in the backpropagation method.
                        \item learning.rate: Learning rate parameter. Notice that we can use a different rate for each neuron.
                }
        \item ADAPTgdwm method:
                \itemize{
                        \item delta: Used in the backpropagation method.
                        \item learning.rate: Learning rate parameter. Notice that we can use a different rate for each neuron.
                        \item momentum: Momentum constant used in the backpropagation with momentum learning criterium.
                        \item former.weight.change: Last increment in the weight parameters. Used by the momentum training technique.
                        \item former.bias.change: Last increment in the bias parameter. Used by the momentum training technique.
                }
        \item BATCHgd method: 
                \itemize{
                        \item delta: Used in the backpropagation method.
                        \item learning.rate: Learning rate parameter. Notice that we can use a different rate for each neuron.
                        \item sum.delta.x: Used as an acumulator of the changes to apply to the weight parameters in the batch training.
                        \item sum.delta.bias: Used as an acumulator of the changes to apply to the bias parameters in the batch training.
                }
        \item BATCHgdwm method:
                \itemize{
                        \item delta: Used in the backpropagation method.
                        \item learning.rate: Learning rate parameter. Notice that we can use a different rate for each neuron.
                        \item sum.delta.x: Used as an acumulator of the changes to apply to the weight parameters in the batch training.
                        \item sum.delta.bias: Used as an acumulator of the changes to apply to the bias parameters in the batch training.
                        \item momentum: Momentum constant used in the backpropagation with momentum learning criterium.
                        \item former.weight.change: Last increment in the weight parameters. Used by the momentum training technique.
                        \item former.bias.change: Last increment in the bias parameter. Used by the momentum training technique.
                }
}}
}

\value{\emph{init.MLPneuron} returns a single neuron. Mainly used to create a neural network object.}
\author{
Manuel Castejón Limas. 	            \email{manuel.castejon@gmail.com}\cr
Joaquin Ordieres Meré.	            \email{j.ordieres@upm.es}\cr
Ana González Marcos.                \email{ana.gonzalez@unirioja.es} \cr
Alpha V. Pernía Espinoza.           \email{alpha.pernia@unirioja.es}\cr
Francisco Javier Martinez de Pisón. \email{fjmartin@unirioja.es}\cr
Fernando Alba Elías.                \email{fernando.alba@unavarra.es}\cr
}

\seealso{
\code{\link{newff}}, \code{\link{random.init.MLPnet}}, \code{\link{random.init.MLPneuron}}, \code{\link{select.activation.function}} , \code{\link{init.MLPneuron}}
}
\keyword{neural}

