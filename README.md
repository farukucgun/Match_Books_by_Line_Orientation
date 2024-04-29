# Match_Books_by_Line_Orientation

* Use Canny edge detector to find the edges.

* Find straight line segments in the edge detected output using Hough transform.

* Compute line orientation histograms.

* Rotated book histograms can be considered as shifted versions of the corresponding original book histograms. Shift the number of bins in the histogram and use the Euclidean distance between the shifted histograms and the original book histograms to match the books.

* Minimum Euclidean distance points at a reasonable match.

* The angle of rotation can be deduced from the number of bins we had to shift.