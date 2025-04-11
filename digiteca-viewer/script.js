document.addEventListener('DOMContentLoaded', () => {
    const prevButton = document.getElementById('prev-button');
    const nextButton = document.getElementById('next-button');
    const itemInfo = document.getElementById('item-info');
    const frontImage = document.getElementById('front-image');
    const backImage = document.getElementById('back-image');
    const frontImageContainer = document.getElementById('front-image-container');
    const backImageContainer = document.getElementById('back-image-container');
    const frontSpyglass = document.getElementById('front-spyglass');
    const backSpyglass = document.getElementById('back-spyglass');
    const metadataDisplay = document.getElementById('metadata-display');

    const IIIF_BASE_URL = "https://iiif.itatti.harvard.edu/iiif/2/";
    // Paths relative to the viewer's HTML file (digiteca-viewer/index.html)
    const TRANSCRIPTIONS_DIR = '../transcriptions/'; // Go up one level, then into transcriptions
    const INDEX_PATH = 'index.json'; // Index file is now local to the viewer HTML

    let transcriptionFiles = []; // Will hold the list of JSON filenames from index.json
    let currentIndex = -1;

    // --- Helper function to check for empty values recursively ---
    function isEmptyRecursive(value) {
        if (value === null || value === undefined || value === '') {
            return true;
        }
        if (Array.isArray(value)) {
            if (value.length === 0) {
                return true;
            }
            // Check if all elements in the array are recursively empty
            return value.every(item => isEmptyRecursive(item));
        }
        // Check for empty object AFTER checking array
        if (typeof value === 'object' && Object.keys(value).length === 0) {
            return true;
        }
        if (typeof value === 'object') {
            // Check if all properties in the object are recursively empty
            return Object.keys(value).every(key => isEmptyRecursive(value[key]));
        }
        // If it's a non-empty primitive (number, boolean, non-empty string)
        return false;
    }
    // --- End of helper function ---


    // --- Modified renderMetadata function to hide empty fields ---
    function renderMetadata(data, parentElement, level = 0) {
        for (const key in data) {
            if (data.hasOwnProperty(key)) {
                const value = data[key];

                // *** Check if the value is empty before rendering anything for this key ***
                if (!isEmptyRecursive(value)) {
                    // Create container div for this key-value pair or key-object pair
                    const entryDiv = document.createElement('div');
                    entryDiv.classList.add(`level-${level}`);

                    // Display the key (bold)
                    const strong = document.createElement('strong');
                    strong.textContent = `${key}:`;
                    entryDiv.appendChild(strong);

                    // Append this entry's div (key) to the parent element *only if value is not empty*
                    parentElement.appendChild(entryDiv);

                    // Now handle the non-empty value
                    if (typeof value === 'object' && value !== null) { // Already checked for null/empty object/array above
                        if (Array.isArray(value)) {
                            // Handle non-empty arrays: Render non-empty items indented under the parent
                            value.forEach((item) => {
                                // *** Check if the array item itself is empty ***
                                if (!isEmptyRecursive(item)) {
                                    const itemDiv = document.createElement('div');
                                    itemDiv.classList.add(`level-${level + 1}`, 'array-item');
                                    if (typeof item === 'object' && item !== null) {
                                        // If array item is a non-empty object, render its structure
                                        renderMetadata(item, itemDiv, 0); // Render relative to itemDiv
                                    } else {
                                        // Otherwise, display the non-empty primitive value
                                        itemDiv.appendChild(document.createTextNode(item));
                                    }
                                    // Append the item div (containing value or nested structure) to the main parent
                                    parentElement.appendChild(itemDiv);
                                }
                            });
                        } else {
                            // Handle non-empty nested objects: Recurse, rendering contents indented under the parent key
                            // The recursive call will handle hiding empty fields within the nested object
                            renderMetadata(value, parentElement, level + 1); // Pass same parent, increase level
                        }
                    } else {
                        // Handle non-empty simple values: Append value to the key's div
                        entryDiv.appendChild(document.createTextNode(` ${value}`)); // Add space, value is guaranteed non-null/non-empty string here
                    }
                }
                // *** If isEmptyRecursive(value) is true, do nothing - skip rendering this key/value ***
            }
        }
    }
    // --- End of modified renderMetadata function ---


    async function loadItem(index) {
        if (index < 0 || index >= transcriptionFiles.length) {
            console.error("Index out of bounds:", index);
            return;
        }
        currentIndex = index;
        const jsonFilename = transcriptionFiles[index];
        const jsonPath = `${TRANSCRIPTIONS_DIR}${jsonFilename}`;

        itemInfo.textContent = `Loading item ${index + 1} / ${transcriptionFiles.length}...`;
        metadataDisplay.innerHTML = '<p>Loading metadata...</p>'; // Clear display initially
        frontImage.src = ''; // Clear images while loading
        backImage.src = '';

        try {
            // Fetch the specific transcription JSON
            const response = await fetch(jsonPath);
            if (!response.ok) {
                throw new Error(`Transcription file not found or invalid (Status: ${response.status}) - ${jsonPath}`);
            }
            const jsonData = await response.json();

            // Extract data needed for IIIF URLs and display
            const barcode = jsonData.barcode || 'N/A';
            const boxBarcode = jsonData.box_barcode;
            const rectoFilenameTif = jsonData.recto_filename;
            const versoFilenameTif = jsonData.verso_filename;

            itemInfo.textContent = `Item ${index + 1} / ${transcriptionFiles.length} (Barcode: ${barcode})`;

            // Validate required fields for IIIF URL construction
            if (!boxBarcode || !rectoFilenameTif || !versoFilenameTif) {
                 throw new Error(`Missing required fields (box_barcode, recto_filename, verso_filename) in ${jsonFilename}`);
            }

            // Construct IIIF URLs - replacing .tif with .jpg
            const rectoFilenameJpg = rectoFilenameTif.replace('.tif', '.jpg');
            const versoFilenameJpg = versoFilenameTif.replace('.tif', '.jpg');
            const iiifIdentifierRecto = `digiteca!${boxBarcode}!${rectoFilenameJpg}`;
            const iiifIdentifierVerso = `digiteca!${boxBarcode}!${versoFilenameJpg}`;

            // Load full resolution images directly into the img tags
            const displayUrlRecto = `${IIIF_BASE_URL}${encodeURIComponent(iiifIdentifierRecto)}/full/full/0/default.jpg`;
            const displayUrlVerso = `${IIIF_BASE_URL}${encodeURIComponent(iiifIdentifierVerso)}/full/full/0/default.jpg`;

            // Full size URL is the same as display URL now, used for spyglass background
            const fullUrlRecto = displayUrlRecto;
            const fullUrlVerso = displayUrlVerso;

            frontImage.src = displayUrlRecto;
            backImage.src = displayUrlVerso;

            // Setup spyglass after images potentially load (or use onload event)
            setupSpyglass(frontImageContainer, frontImage, frontSpyglass, fullUrlRecto);
            setupSpyglass(backImageContainer, backImage, backSpyglass, fullUrlVerso);

            // Render only the 'annotations' part of the metadata
            if (jsonData.annotations) {
                 metadataDisplay.innerHTML = ''; // Clear "Loading..." message before rendering
                 renderMetadata(jsonData.annotations, metadataDisplay); // Initial call
            } else {
                 metadataDisplay.innerHTML = '<p>No "annotations" field found in JSON data.</p>';
            }

        } catch (error) {
            console.error("Error loading item:", index, jsonFilename, error);
            itemInfo.textContent = `Error loading item ${index + 1}`;
            metadataDisplay.innerHTML = `<p>Error loading data for ${jsonFilename}: ${error.message}</p>`;
        }

        updateNavButtons();
    }

    function updateNavButtons() {
        prevButton.disabled = currentIndex <= 0;
        nextButton.disabled = currentIndex >= transcriptionFiles.length - 1;
    }

    function setupSpyglass(container, image, spyglass, fullImageUrl) {
        // Ensure image is loaded to get correct dimensions, or use container dimensions
        const zoomLevel = .25; // Set zoom to 100% (no magnification) - NOTE: 1.0 means spyglass shows image at native resolution

        spyglass.style.backgroundImage = `url('${fullImageUrl}')`;

        container.onmousemove = (e) => {
            spyglass.style.display = 'block';

            // Calculate cursor position relative to the container
            const rect = container.getBoundingClientRect();
            let x = e.clientX - rect.left;
            let y = e.clientY - rect.top;

            // Use the container's dimensions for positioning calculations
            const containerWidth = container.offsetWidth;
            const containerHeight = container.offsetHeight;

            // Use the image's natural dimensions for background scaling
            const naturalWidth = image.naturalWidth;
            const naturalHeight = image.naturalHeight;

            // If natural dimensions aren't available yet, bail out
            if (!naturalWidth || !naturalHeight) {
                spyglass.style.display = 'none'; // Hide if we can't calculate
                return;
            }

            // Prevent spyglass from moving outside the image boundaries
            const spyglassWidth = spyglass.offsetWidth;
            const spyglassHeight = spyglass.offsetHeight;

            // Center spyglass on cursor
            let spyX = x - spyglassWidth / 2;
            let spyY = y - spyglassHeight / 2;

            spyglass.style.left = `${spyX}px`;
            spyglass.style.top = `${spyY}px`;

            // Calculate background position for zoom effect
            // Need ratio of cursor position relative to the *container*
            const ratioX = x / containerWidth;
            const ratioY = y / containerHeight;

            // Calculate background size based on zoom level and *natural* image size
            const bgWidth = naturalWidth * zoomLevel;
            const bgHeight = naturalHeight * zoomLevel;
            spyglass.style.backgroundSize = `${bgWidth}px ${bgHeight}px`;

            // Calculate background position for the zoom effect
            // We want the point under the cursor (x, y) on the *displayed* image
            // to correspond to the same point on the *background* image,
            // but scaled by the zoom level. Then, we shift it so that this point
            // appears at the center of the spyglass.

            // Calculate background position
            // Offset is negative ratio * (zoomed natural size - spyglass size)
            // Use the ratio relative to the container size
            let bgPosX = -(ratioX * bgWidth - spyglassWidth / 2);
            let bgPosY = -(ratioY * bgHeight - spyglassHeight / 2);

            spyglass.style.backgroundPosition = `${bgPosX}px ${bgPosY}px`;
        };

        container.onmouseleave = () => {
            spyglass.style.display = 'none';
        };
         container.onmouseenter = () => {
             // Pre-set background image when entering container
             spyglass.style.backgroundImage = `url('${fullImageUrl}')`;
         };
    }


    async function initialize() {
        try {
            const response = await fetch(INDEX_PATH);
            if (!response.ok) {
                throw new Error(`Failed to fetch index file (Status: ${response.status}) - ${INDEX_PATH}`);
            }
            transcriptionFiles = await response.json();
            console.log(`Loaded index with ${transcriptionFiles.length} files.`);

            if (transcriptionFiles.length > 0) {
                loadItem(0); // Load the first item
            } else {
                itemInfo.textContent = "No transcriptions found.";
                metadataDisplay.innerHTML = '<p>No transcription files listed in index.json.</p>';
            }
        } catch (error) {
            console.error("Error initializing viewer:", error);
            itemInfo.textContent = "Error loading index.";
            metadataDisplay.innerHTML = `<p>Could not load ${INDEX_PATH}. Did you run the Python script to generate it?</p>`;
        }


        prevButton.addEventListener('click', () => {
            if (currentIndex > 0) {
                loadItem(currentIndex - 1);
            }
        });

        nextButton.addEventListener('click', () => {
            if (currentIndex < transcriptionFiles.length - 1) {
                loadItem(currentIndex + 1);
            }
        });
    }

    initialize();
});
